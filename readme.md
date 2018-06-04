## Visualisation of convolutions in keras models

Actually, I was wondering how the kernels in a convolutional network really look like.  As it is difficult to visualize an e.g. 3x3x128 dimensional kernel directly, the output of each kernel for a given picture can be visualized.  My ambitions resulted in 2 function which either show all (or a subset) kernels for a given image or show a small number of kernels for a number of given images.  

The convolutions displayed come from 

> trunc_model = Model(inputs=model.input, outputs=model.get_layer(conv_layer_name).output)
>
> result = trunc_model.predict(batch_image)

The two methods created are. 
* from vis import vis_conv
vis_conv.display_convolutions_per_image(...)
vis_conv.display_convolutions_per_filter(...)

The code is in the vis_conv package and produces images like:
 

