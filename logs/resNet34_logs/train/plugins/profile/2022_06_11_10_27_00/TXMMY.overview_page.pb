?	???*??U@???*??U@!???*??U@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???*??U@~T?~O?@1^I?\?M@A?`??q??I?u???4@*	?????P?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator&䃞ͪ??!??D??X@)&䃞ͪ??1??D??X@:Preprocessing2F
Iterator::ModelΪ??V???!      Y@)?q????o?1??~j????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism\ A?c???!'??a0?X@){?G?zd?1;w?x*???:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapGr?????!??7p?X@)-C??6Z?1?'?q?G??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?23.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?q7l?"@@QG?ɰ?P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~T?~O?@~T?~O?@!~T?~O?@      ??!       "	^I?\?M@^I?\?M@!^I?\?M@*      ??!       2	?`??q???`??q??!?`??q??:	?u???4@?u???4@!?u???4@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?q7l?"@@yG?ɰ?P@?"d
8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterADj?(???!ADj?(???0"d
8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?^]????!j??n???0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??
?ˇ??!f?4?)2??0"d
8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?;(?j???!X???D???0"d
8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilters??y??!?????x??0"d
8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???jww??!?????'??0"b
7gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput???.l??!ڞ??????0"d
8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?5J}????!3BZf7???0"e
9gradient_tape/model/conv2d_15/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter ??????!U????0"d
8gradient_tape/model/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter+??k?T??!o{??;??0Q      Y@Y??????a??Gu??X@q??L?i?T@y????'l?"?
both?Your program is POTENTIALLY input-bound because 8.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?23.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?83.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 