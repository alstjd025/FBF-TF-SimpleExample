#include "tensorflow/lite/tf_scheduler.h"

#define SCHEDULER_SOCK "/home/nvidia/FBF-TF-SimpleExample/new_bench/sock/scheduler"

int main(){
  tflite::TfScheduler scheduler(SCHEDULER_SOCK);
  scheduler.Work();

  return 0;
}