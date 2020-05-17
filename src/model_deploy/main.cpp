#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "uLCD_4DGL.h"
#include <cmath>
#include "DA7212.h"

#define bufferLength (32)
#define signalLength (1024)

DA7212 audio;
Serial pc(USBTX, USBRX);
InterruptIn button(SW2);
DigitalIn  Switch(SW3);
DigitalOut green_led(LED2);
uLCD_4DGL uLCD(D1, D0, D2);

EventQueue DNNqueue(32 * EVENTS_EVENT_SIZE);
Thread DNNthread(osPriorityNormal,80*1024/*120K stack size*/);

int num = 0;
int gesture;
int song = 0;
int mode = 0;
int push = 0;
int state = 0;
int print = 0;
int nowplay = 0;
int main_page = 0;
int serialCount = 0;
float song_note[42];
int change_mode = 0;
int change_song = 0;
float noteLength[42];
int change_mode_to = 0;
int16_t waveform[kAudioTxBufferSize];
char SerialInBuffer[bufferLength];
char list[3][15]={"Little star", "YAMAHA", "Little bee"};

void loadSignal(void)
{
  green_led = 0;
  int i = 0;
  serialCount = 0;
  audio.spk.pause();
  serialCount =0;
  while(i < 42)
  {
    if(pc.readable())
    {
      SerialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 5)
      {
        SerialInBuffer[serialCount] = '\0';
        song_note[i] = (float) atof(SerialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
  i = 0;
  serialCount =0;
  while(i < 42)
  {
    if(pc.readable())
    {
      SerialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 5)
      {
        SerialInBuffer[serialCount] = '\0';
        noteLength[i] = (float) atof(SerialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
  green_led = 1;
}


void playNote(float freq[])
{
    float frequency =  freq[num];
    for(int j = 0; (j < kAudioSampleFrequency / kAudioTxBufferSize)&& !push; ++j)
    {
      for (int i = 0; i < kAudioTxBufferSize; i++)
      {
      waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency /( 500*frequency))) * ((1<<16) - 1));
      }
      audio.spk.play(waveform, kAudioTxBufferSize);
    }     
}

void change_mode()
{
  if(push ==0)
  {
    push=1;
  }
  else 
    push=0;
  
  audio.spk.pause();
  print =1;
  main_page =0;

}

int PredictGesture(float* output)
{
  static int continuous_count = 0;
  static int last_predict = -1;
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }
  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]){
    return label_num;
  }
  continuous_count = 0;
  last_predict = -1;
  return this_predict;
}

void DNN()
{
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  bool should_clear_buffer = false;
  bool got_data = false;
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  static tflite::MicroOpResolver<6> micro_op_resolver;

  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,

                               tflite::ops::micro::Register_MAX_POOL_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,

                               tflite::ops::micro::Register_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,

                               tflite::ops::micro::Register_FULLY_CONNECTED());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,

                               tflite::ops::micro::Register_SOFTMAX());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(),1);

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  interpreter->AllocateTensors();

  TfLiteTensor* model_input = interpreter->input(0);

  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelnum) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");

  while (true) {

    got_data = ReadAccelerometer(error_reporter, model_input->data.f,

                                 input_length, should_clear_buffer);

    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }
    gesture = PredictGesture(interpreter->output(0)->data.f);
    should_clear_buffer = gesture < label_num;
  }
}

int main(int argc, char* argv[])
{
  green_led =1;
  DNNthread.start(DNN);
  button.rise(&change_mode); 
  audio.spk.pause();
  
  while(true){
    if(push){
      audio.spk.pause(); 
      if(change_mode_to == 0)
      {
        if(print){
          if(mode == 0){
            if(nowplay > 1)
              song = nowplay -1;
            else 
              song =2;
            uLCD.cls();
            uLCD.printf("backward song\n\n\n\n");
            uLCD.printf("%s",list[song]);
          }
          
          if(mode == 1){
            if(nowplay < 2)
              song = nowplay +1;
            else 
              song =0;
            uLCD.cls();
            uLCD.printf("forward song\n\n\n\n");
            uLCD.printf("%s",list[song]);
          }
            
          if(mode == 2){
            uLCD.cls();
            uLCD.printf("change songs\n\n\n\n");
            uLCD.printf("%s\n",list[0]);
            uLCD.printf("%s\n",list[1]);
            uLCD.printf("%s\n",list[2]);
          }            
        print=0;  
        }

        if(gesture == 0){
          state = 1;
          if(mode < 1)
            mode = 3;
          else
          {
            mode = mode-1;
          }
          change_song = 0;
        }
        if(gesture == 1){
          state = 1;
          if(mode > 2)
            mode = 0;
          else
          {
            mode = mode+1;
          }
          change_song = 0;
        }
        if(mode == 0){
          if(state){
          
            if(mode == 0){
              if(nowplay > 0)
                song = nowplay -1;
              else 
                song = 2;  
            }
            uLCD.cls();
            state=0;
            uLCD.printf("backward song\n\n\n\n");
            uLCD.printf("%s",list[song]); 
            main_page =0;
          }
          uLCD.color(GREEN);
        }
        if(mode == 1){
          if(state){
            if(mode == 1){
              if(nowplay < 2)
                song = nowplay +1;
              else 
                song =0;  
            }
            uLCD.cls();
            state = 0;
            uLCD.printf("forward song\n\n\n\n");
            uLCD.printf("%s",list[song]);
            main_page =0;
          }
          uLCD.color(GREEN);
        }
        if(mode == 2){
          if(state){
            uLCD.cls();
            state=0;
            uLCD.printf("change songs\n\n\n\n");
            uLCD.printf("%s\n",list[0]);
            uLCD.printf("%s\n",list[1]);
            uLCD.printf("%s\n",list[2]);
            main_page =0;
          }
          uLCD.color(GREEN);

        if(Switch == 0){
            change_mode_to = 1;
        }
      }  
    }
    if(change_mode_to == 1){
        if(gesture == 0){
          change_mode = 0;
          if(song < 1)
            song = 2;
          else
          {
            song=song-1;
          }
        }
        if(gesture == 1){
          change_mode = 0;
          if(song > 1)
            song = 0;
          else
          {
            song = song+1;
          }
        }
        if(change_mode == 0){
          uLCD.cls();
          change_mode = 1;

          uLCD.color(GREEN);  
          if(mode == 2) 
            uLCD.printf("change songs\n\n\n\n");
          else 
            uLCD.printf("backward song\n\n\n\n");
          if(song == 0){
            uLCD.color(WHITE);
            uLCD.printf("%s\n",list[0]);
            uLCD.color(GREEN);
            uLCD.background_color(BLACK);

            uLCD.printf("%s\n",list[1]);
            uLCD.printf("%s\n",list[2]);
          }
          if(song == 1){
            uLCD.color(GREEN);
            uLCD.background_color(BLACK);            
            uLCD.printf("%s\n",list[0]);         
            uLCD.color(WHITE);
            uLCD.printf("%s\n",list[1]);
            uLCD.color(GREEN);
            uLCD.background_color(BLACK);            
            uLCD.printf("%s\n",list[2]); 
          }
          if(song == 2){
            uLCD.color(GREEN);       
            uLCD.printf("%s\n",list[0]);
            uLCD.printf("%s\n",list[1]);
            uLCD.color(WHITE);         
            uLCD.printf("%s\n",list[2]);
          }
        }
    }
  }
  }
}