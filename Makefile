TARGET = unit_simple
OBJECTS = unit_simple.o unit.o unit_handler.o
SRCS = unit_simple.cc unit.cc unit_handler.cc
INC = -I/home/xavier/tensorflow\
		-I/home/xavier/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include\
		-I/home/xavier/tensorflow/tensorflow/lite/tools/make/downloads/absl
LIBS = -lopencv_gapi\
		-ltensorflow-lite\
		-lflatbuffers /lib/aarch64-linux-gnu/libdl.so.2\
		-lopencv_stitching -lopencv_alphamat -lopencv_aruco -lopencv_bgsegm\
		-lopencv_bioinspired -lopencv_ccalib -lopencv_cudabgsegm -lopencv_cudafeatures2d\
		-lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_dnn_objdetect\
		-lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype\
		-lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform\
		-lopencv_line_descriptor -lopencv_mcc -lopencv_quality -lopencv_rapid -lopencv_reg\
		-lopencv_rgbd -lopencv_saliency -lopencv_sfm -lopencv_stereo -lopencv_structured_light\
		-lopencv_phase_unwrapping -lopencv_superres -lopencv_cudacodec -lopencv_surface_matching\
		-lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_videostab\
		-lopencv_cudaoptflow -lopencv_optflow -lopencv_cudalegacy -lopencv_videoio -lopencv_cudawarping\
		-lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect\
		-lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann\
		-lopencv_xphoto -lopencv_photo -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_imgproc\
		-lopencv_cudaarithm -lopencv_core -lopencv_cudev -lpthread\
		/home/xavier/tensorflow/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so\
		/usr/lib/aarch64-linux-gnu/libEGL.so\
		/usr/lib/aarch64-linux-gnu/libGL.so /usr/lib/aarch64-linux-gnu/libGLESv2.so
LIBPATH = -L/home/xavier/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib\
			-L/home/xavier/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build
CC = g++


all : unit

unit.o : unit.h unit.cc
		g++ -c -o unit.o unit.cc $(INC) 

unit_handler.o : unit_handler.h unit_handler.cc unit.h
		g++ -c -o unit_handler.o unit_handler.cc $(INC) 

unit_simple.o : unit_simple.cc unit_handler.h 
		g++ -c -o unit_simple.o unit_simple.cc $(INC) 

unit : unit_simple.o unit_handler.o unit.o 
		g++ -o unit unit_simple.o unit_handler.o unit.o $(INC) $(LIBPATH) $(LIBS) 

clean : 
		rm -f *.o \
    	rm -f unit
