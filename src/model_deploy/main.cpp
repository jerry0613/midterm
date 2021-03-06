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

#include "mbed.h"
#include <cmath>
#include "DA7212.h"
#include "uLCD_4DGL.h"

#define bufferLength (32)
#define signalLength 84
#define mode_num 5
#define name_num 4
#define do 261
#define re 294
#define mi 330
#define fa 349
#define so 392
#define la 440
#define si 494


DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
uLCD_4DGL uLCD(D1, D0, D2);
InterruptIn sw2(SW2);
InterruptIn sw3(SW3);
Serial pc(USBTX, USBRX);
DigitalOut green_led(LED2);

int mode = 0; // 0: play song, 1: list
int serialCount = 0, score = 0;
int cur = 0, cur1 = 0, flag = 1, cur2 = 0, first = 0;
char serialInBuffer[bufferLength];
char *change[5] = {"Song1", "Song2", "Song3", "Song4", "Song5"};
char *name[name_num] = {"Song1", "Song2", "Song3", "Song4"};
char *state[mode_num] = {"Forward", "Backward", "Song selection", "Change song", "Back"};
char temp[20];

int song[4][42] = {
  {261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261}, 

  {si, la, so, la, si, si, si,
   la, la, la, si, si, si, si, 
   la, so, la, si, si, si, la,
   la, si, la, so}, 
  {}, 
  {}
  };

int noteLength[4][42] = {
  {1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2},

  {1, 1, 1, 1, 1, 1, 2, 
   1, 1, 2, 1, 1, 2, 1, 
   1, 1, 1, 1, 1, 2, 1, 
   1, 1, 1, 2},
  {},
  {}
  };

int note[42] = {
  0, 0, 1, 0, 0, 1, 0,
  0, 2, 0, 0, 1, 0, 0,
  2, 0, 0, 2, 0, 0, 2,
  0, 0, 1, 0, 0, 1, 0,
  0, 1, 0, 0, 2, 0, 0,
  1, 0, 0, 2, 0, 0, 0
};


void playNote(int freq)
{
  for(int k = 0; k < kAudioTxBufferSize; k++)
  {
    waveform[k] = (int16_t) (sin((double)k * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}

int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.7) this_predict = i;
  }

  // No gesture was detected above the threshold

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

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void mode_1() {
  mode = 1;
  flag = 1;
}

void mode_0() {
  flag++;
  if (mode == 3) {
    mode = 4;
    name[cur] = change[cur2];
  } else if (cur1 == 3) {
    mode = 3;
  }
  else if (mode == 2 || cur1 != 2) {
    mode = 0;
    if (cur1 == 0) {
        if (cur < name_num - 1)
            cur++;
        else
            cur = 0;
    } else if (cur1 == 1) {
        if (cur == 0)
            cur = name_num - 1;
        else
        {
            cur--;
        }
    }
  }
  else
    mode = 2;
}

constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  bool should_clear_buffer = false;
  bool got_data = false;
  int gesture_index;
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  static tflite::MicroOpResolver<6> micro_op_resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;
  TfLiteTensor* model_input = interpreter->input(0);
  int input_length = model_input->bytes / sizeof(float);
  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

void gesture() {
  while(1) {
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

    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    should_clear_buffer = gesture_index < label_num;
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
    }
  }
}

int main(int argc, char* argv[]) { 
   
  green_led = 1;
  sw2.fall(&mode_1);
  sw3.fall(&mode_0);

  /*constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);*/
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

 // static tflite::MicroOpResolver<6> micro_op_resolver;
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
                             tflite::ops::micro::Register_RESHAPE(), 1);

  /*static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;*/

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  //TfLiteTensor* model_input = interpreter->input(0);

  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  //int input_length = model_input->bytes / sizeof(float);

  //TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");

  Thread t(osPriorityNormal, 100 * 1024);
  t.start(gesture);

  while (true) {
      
      /*got_data = ReadAccelerometer(error_reporter, model_input->data.f,
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

      gesture_index = PredictGesture(interpreter->output(0)->data.f);

      should_clear_buffer = gesture_index < label_num;
      if (gesture_index < label_num) {
        error_reporter->Report(config.output_message[gesture_index]);
      }*/

      if (mode == 0) {
        if (flag == 1) {
          uLCD.cls();
          uLCD.printf("Song NO.%d   Score\n", cur+1);
          score = 0;
          uLCD.printf("%s        %d\n", name[cur], score);
          flag = 0;
          uLCD.line(0, 20, 200, 20, GREEN);
          uLCD.line(0, 120, 200, 120, GREEN);
        }

        score = 0;
        int det = 0;
        
        for(int i = 0; i < 42 && mode == 0; i++)
        {
          int length = noteLength[cur][i];
          while(length-- && mode == 0)
          { 
            
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize && mode == 0; ++j)
              playNote(song[cur][i]);  

            if (length < 1)
              playNote(0);
            det = 0;

            for (int k = 1; k < 21 && length < 1 && mode == 0; k++) {
                if (note[i] == 2)
                    uLCD.line(30, 20+k*5, 100, 20+k*5, RED);
                else if (note[i] == 1)
                    uLCD.line(30, 20+k*5, 100, 20+k*5, BLUE);
                
                if (k > 1)
                  uLCD.line(30, 20+k*5-5, 100, 20+k*5-5, 0);
  
                uLCD.locate(0, 5);
                uLCD.printf("%d", gesture_index);

                if (!det && k > 17) {
                  if ((gesture_index == 1 && note[i] == 1) ||
                      (gesture_index == 0 && note[i] == 2)) {
                    score += 10;
                    uLCD.locate(0, 1);
                    uLCD.printf("%s        %d\n", name[cur], score);
                    uLCD.line(30, 20+k*5, 100, 20+k*5, 0);
                    det++;
                  }
                }
                if (k == 20)
                  uLCD.line(30, 120, 100, 120, GREEN);
            }
          }
        }
        mode = 1;
        flag = 1;
    }

    if (mode == 1) { // mode 1
        if (flag == 1) {
          /*for (int i = 0; i < 20; i++)
            pc.printf("%d ", temp[i]);*/
          uLCD.cls();
          uLCD.printf("Your score %d", score);
          uLCD.locate(5, 5);
          uLCD.printf("\n%s\n", state[cur1]);
          flag = 0;
          playNote(0);
        }
        if (gesture_index == 0) {
          if (cur1 > 0)
            cur1--;
          else
            cur1 = mode_num - 1;
          uLCD.cls();
          uLCD.locate(5, 5);
          uLCD.printf("\n%s\n", state[cur1]);
        } else if (gesture_index == 1) {
          if (cur1 < mode_num - 1)
            cur1++;
          else
            cur1 = 0;
          uLCD.cls();
          uLCD.locate(5, 5);
          uLCD.printf("\n%s\n", state[cur1]);
        } 
    }

    if (mode == 2) {  // mode 2
        if (flag == 1) {
            flag = 0;
            uLCD.cls();
            uLCD.locate(5, 5);
            uLCD.printf("\nSong selection\n");
            uLCD.printf("\nSong NO.%d\n", cur+1);
            uLCD.printf("\n%s\n", name[cur]);
        }
        if (gesture_index == 0) {
          if (cur > 0)
            cur--;
          else
            cur = name_num - 1;
          uLCD.cls();
          uLCD.locate(5, 5);
          uLCD.printf("\nSong selection\n");
          uLCD.printf("\nSong NO.%d\n", cur+1);
          uLCD.printf("\n%s\n", name[cur]);

        } else if (gesture_index == 1) {
          if (cur < name_num - 1)
            cur++;
          else
            cur = 0;
          uLCD.cls();
          uLCD.locate(5, 5);
          uLCD.printf("\nSong selection\n");
          uLCD.printf("\nSong NO.%d\n", cur+1);
          uLCD.printf("\n%s\n", name[cur]);
        } 
    }

    if (mode == 3) {
        if (flag == 1) {
            flag = 0;
            uLCD.cls();
            uLCD.locate(5, 5);
            uLCD.printf("\nExchang song\n");
            uLCD.printf("\n%s\n", change[cur2]);
        }
        if (gesture_index == 0) {
          if (cur2 > 0)
            cur2--;
          else
            cur2 = 4;
          uLCD.cls();
          uLCD.locate(5, 5);
          uLCD.printf("\nExchang song\n");
          uLCD.printf("\n%s\n", change[cur2]);

        } else if (gesture_index == 1) {
          if (cur2 < 4)
            cur2++;
          else
            cur2 = 0;
          uLCD.cls();
          uLCD.locate(5, 5);
          uLCD.printf("\nExchang song\n");
          uLCD.printf("\n%s\n", change[cur2]);
        } 
    }

    if (mode == 4) {
      uLCD.cls();
      uLCD.locate(5, 5);
      uLCD.printf("\nLoading song\n");
      
      green_led = 0;
      int i = 0;
      serialCount = 0;

      if (first == 0) {
        pc.getc();
        pc.getc();
      }
    

      if (first)
       i += 2;
      //int j = 0;
      first = 1;

      while(i < 44)
      {
        if(pc.readable())
        {
          serialInBuffer[serialCount] = pc.getc();
          //temp[j] = serialInBuffer[serialCount];
          //j++;
          
          serialCount++;

          if(serialCount == 3)
          {
            serialInBuffer[serialCount] = '\0';
            
            if (i > 1)
              song[cur][i-2] = (int) atoi(serialInBuffer);
            
            serialCount = 0;
            i++;
          }
        }
      }

      i = 0;
      serialCount = 0;

      while(i < 42)
      {
        if(pc.readable())
        {
          serialInBuffer[serialCount] = pc.getc();
      
          serialCount++;

          if(serialCount == 3)
          {
            serialInBuffer[serialCount] = '\0';
            noteLength[cur][i] = (int) atoi(serialInBuffer);
            noteLength[cur][i] -= 100;
            serialCount = 0;
            i++;
          }
        }
      }

      green_led = 1;
      mode = 0;
      flag = 1;
    }
    
  }
}