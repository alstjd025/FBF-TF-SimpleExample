unit_simple : unit_simple.cc
	g++ -o unit_simple unit_simple.cc -I/usr/include/opencv4\
		-pthread -g -I/home/odroid/tensorflow\
		-I/home/odroid/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include\
		-I/home/odroid/tensorflow/tensorflow/lite/tools/make/downloads/absl\
		-L/home/odroid/tensorflow/tensorflow/lite/tools/make/gen/bbb_armv7l/lib\
		-L/home/odroid/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/BUILD\
		-L/usr/lib/arm-linux-gnueabihf\
		-lopencv_gapi\
		-ltensorflow-lite\
		-lflatbuffers /lib/arm-linux-gnueabihf/libdl.so.2\
		-lopencv_stitching -lopencv_aruco -lopencv_bgsegm\
		-lopencv_bioinspired -lopencv_ccalib\
		-lopencv_dnn_objdetect\
		-lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype\
		-lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform\
		-lopencv_line_descriptor -lopencv_mcc -lopencv_quality -lopencv_rapid -lopencv_reg\
		-lopencv_rgbd -lopencv_saliency -lopencv_structured_light\
		-lopencv_phase_unwrapping -lopencv_superres -lopencv_surface_matching\
		-lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_videostab\
		-lopencv_optflow -lopencv_videoio\
		-lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect\
		-lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann\
		-lopencv_xphoto -lopencv_photo -lopencv_imgproc\
		-lopencv_core\
		/home/odroid/tensorflow/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so\
		/usr/lib/arm-linux-gnueabihf/libGL.so\
		/usr/lib/arm-linux-gnueabihf/libEGL.so\
		/usr/lib/arm-linux-gnueabihf/libGLESv2.so\

