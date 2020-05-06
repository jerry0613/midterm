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
#define mode_num 4
#define name_num 4

DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
uLCD_4DGL uLCD(D1, D0, D2);
InterruptIn sw2(SW2);
InterruptIn sw3(SW3);
Serial pc(USBTX, USBRX);
DigitalOut green_led(LED2);

int mode = 0; // 0: play song, 1: list
int serialCount = 0;
int cur = 0, cur1 = 0, flag = 1, cur2 = 0;
char serialInBuffer[bufferLength];
char *change[5] = {"Song1", "Song2", "Song3", "Song4", "Song5"};
char *name[name_num] = {"Song1", "Song2", "Song3", "Song4"};
char *state[mode_num] = {"Forward", "Backward", "Song selection", "Change song"};

int song[4][42] = {
  {261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261}, 
  {}, 
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
  {},
  {},
  {}
  };

void playNote(int freq)
{
  for(int k = 0; k < kAudioTxBufferSize; k++)
  {
    waveform[k] = (int16_t) (sin((double)k * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}

// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last predictio
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
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
  flag++;
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
        if (cur < mode_num - 1)
            cur++;
        else
            cur = 0;
    } else if (cur1 == 1) {
        if (cur == 0)
            cur = mode_num - 1;
        else
        {
            cur--;
        }
    }
  }
  else
    mode = 2;
}

int main(int argc, char* argv[]) {
    green_led = 1;
    sw2.fall(&mode_1);
    sw3.fall(&mode_0);

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
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
                             tflite::ops::micro::Register_RESHAPE(), 1);

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
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
      if (mode == 0 && flag == 1) {
          uLCD.cls();
          uLCD.locate(5, 5);
          uLCD.printf("\nSong NO.%d\n", cur+1);
          uLCD.printf("\n%s\n", name[cur]);
          flag = 0;
      }

      if (mode == 0) {
        for(int i = 0; i < 42 && mode == 0; i++)
        {
          int length = noteLength[cur][i];
          while(length-- && mode == 0)
          {
            // the loop below will play the note for the duration of 1s
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize && mode == 0; ++j)
            {
              playNote(song[cur][i]);
              /*uLCD.cls();
              uLCD.locate(5, 5);
              uLCD.printf("\n%d\n", song[i]);*/
            }
            for (int k = 0; k < 100 && length < 1; k++)
              playNote(0);
          }
        }
      }
    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
    }

    if (mode == 1) { // mode 1
        if (flag == 1) {
          uLCD.cls();
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
      audio.spk.pause();
      while(i < 42)
      {
        pc.printf("%d\r\n", cur2);
        if(pc.readable())
        {
          serialInBuffer[serialCount] = pc.getc();
          serialCount++;

          if(serialCount == 3)
          {
            serialInBuffer[serialCount] = '\0';
            song[cur][i] = (int) atoi(serialInBuffer);
            serialCount = 0;
            i++;
          }
        }
      }

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