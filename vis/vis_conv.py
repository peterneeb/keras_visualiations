import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import keras
# from keras.utils import np_utils
import copy
from keras.models import Model, Sequential

from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(mappable):
    """ conveninence method to reduce code when creating an axis as a colorbar.  """
    
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def display_image(img):
    """ conveninence method to display a wide range of images.  Images can be either gray scale (x,y) or color (x,y,3) in format """
    #print('image shape:',img.shape)
    if (img.shape[2] == 1) or len(img.shape) == 2:
        img = img.reshape((img.shape[0],img.shape[1]))
        plt.imshow(img, cmap = 'gray')
    elif len(img.shape) == 3:
        plt.imshow(img)
    elif len(img.shape) == 4:
        plt.imshow(img[0])
    else:
        print('cannot display image with above input dimensions.')
    plt.show()

def get_conv_layer_indices(model):
    """ Identify all layers with an output shape if for in the model, as this are potenially conv layers"""
    result = []
    for i, layer in enumerate(model.layers):
        if  len(layer.output_shape) == 4:
            result.append(i)
    return result

def get_layer_by_name(model,name):
    """ convience method to return the index of a layer identified by name """
    for i, layer in enumerate(model.layers):
        if layer.name == name:
            return layer, i
    print('No layer with the name {} was found.'.format(name))
    return None, None

# used with an older version of keras where the get_output method threw some error 
# def create_trunc_model(model, index):
#     layers = copy.copy(model.layers)
#     layers = layers[:index+1]
#     trunc_model = Sequential(layers)
#     trunc_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return trunc_model


def show_filter(original, output, show_colorbar, use_same_scaling, figsize, max_cols, max_plots, invert_cmap, file_name, dpi):
    """
    Used within the display_convolutions_per_image method to actually make the plots.  
    
    This method is not meant to be imported, the parameters are described in display_convolutions_per_image 
    """
    no_filters = output.shape[2]
    if max_plots != None and no_filters > max_plots:
        no_filters = max_plots
    cols = int(np.floor(np.sqrt(no_filters)))
    rows = int(no_filters / cols)
    if max_cols and max_cols < cols:
        cols = max_cols
        rows = int(np.ceil(no_filters / cols))
    #print('No cols:',cols,'\tNo rows:', rows)
    if invert_cmap:
        cmap = 'gray_r'
    else:
        cmap = 'gray'
    if use_same_scaling:
        vmin = np.min(output[:,:,0:rows*cols])#np.min(output)
        vmax = np.max(output[:,:,0:rows*cols])#np.max(output)
    fig, ax= plt.subplots(rows, cols, sharex='col', sharey='row', figsize=figsize)
    for irow in range(ax.shape[0]):
        for icol in range(ax.shape[1]):
            index = irow*cols+icol
            #print(index)
            if irow*cols+icol+1 > no_filters:
                continue
            if use_same_scaling:
                plot = ax[irow,icol].imshow(output[:,:,index], vmin=vmin, vmax=vmax, cmap=cmap)#, aspect='auto')
            else: 
                plot = ax[irow,icol].imshow(output[:,:,index], cmap=cmap)#, aspect='auto')
            ax[irow,icol].set_title("Kernel "+str(index))
            if show_colorbar:
                if show_colorbar:
                    divider = make_axes_locatable(ax[irow,icol])
                    cax = divider.append_axes("right", size="5%", pad=0.08)
                    fig.colorbar(plot, cax=cax)  
                    #plt.colorbar(plot,ax=ax[irow,icol])
    if file_name != None:
        plt.savefig(file_name, dpi = dpi)


def display_convolutions_per_image(model, conv_layer_name, input_image, figsize = (10,10), show_colorbar = True,
                         max_cols=None, max_plots = None, use_same_scaling = True, invert_cmap = False, file_name = None, dpi = None):
    """
    Display convolutions for a single image at a specified layer.    
    
    

    Parameters
    ----------
    model : Model
        The keras model to be investigated
    conv_layer_name : String 
        the name of the layer, whose output should be displayed.  Can be retrieved with model.summary()
    input_image: array 
        the input_image should have dimensions of either (x,y) for gray scale images of (x,y,3) with np.uint8 values
    show_colorbar: Boolean
        wether a colorbar should be displayed
    use_same_scaling: Boolean
        Set to true, it ensures the same scaling of the colorbar for all displayed convvolutions
    invert_cmap: Boolean
        inverts the display of the convolutions, high values are black now
    file_name: String
        if the file_name is not None the convolutions are saved to file, in which case figsize and dpi become important.
    figsize: Tuple(2)
        The size in inches of the saved figure.  Also influcences the output in a notebook
    dpi: int
        the dpi of the saved image
    max_cols: int
        to improve the readability a max number of columns can be specified, so plots do no become too small
    max_plots:  int
        since the number of kernels can be very large (e.g.512) the plot would become very small.  
        max_plots reduces the number of plots displayed.  

    Returns
    -------
    model
        the truncated model, with the layer specified as the last layer

    """
    layer, conv_layer_index = get_layer_by_name(model, conv_layer_name)
    if layer == None:
        print('No layer with name', conv_layer_name,'was found.')
        return
    output_shape = layer.output_shape
    if len(output_shape) != 4:
        print("Convolutions can only displayed for layers with a 4 dim output tensor.  The last dim is assumed to be \
               the number of channels")
        print("These have the following indices in the supplied model",get_conv_layer_indices(model))
        return
    print('Image shape is',input_image.shape)
    display_image(input_image)
    print('Dimensions of output from layer {} is {}.'.format(conv_layer_name,output_shape))
    no_filters = output_shape[3]
    if max_plots != None and no_filters > max_plots:
        no_filters = max_plots
    print('This results of {} plots with dimensions ({},{}) each.'.format(output_shape[3],output_shape[1],output_shape[2]))
    if max_plots == no_filters:
        print('From these plots {} will be displayed'.format(no_filters))
    #trunc_model = create_trunc_model(model = model, index = conv_layer_index)
    #trunc_model = Model(inputs=model.input, outputs=model.get_output_at(conv_layer_index))
    batch_image = np.expand_dims(input_image, axis = 0)
    #result = trunc_model.predict(x=batch_image, batch_size=1, verbose=0)
    trunc_model = Model(inputs=model.input,
                                 outputs=model.get_layer(conv_layer_name).output)
    result = trunc_model.predict(batch_image)
    
    show_filter(input_image,result[0], use_same_scaling = use_same_scaling, show_colorbar = show_colorbar, figsize = figsize,
                max_cols=max_cols, max_plots = max_plots, invert_cmap = invert_cmap, file_name = file_name, dpi = dpi) 
    
    return trunc_model
    

def show_images(images, output, filter_indices, use_same_scaling, 
                figsize, invert_cmap, dpi, show_colorbar, file_name):
    """
    Used within the display_convolutions_per_filter method to actually make the plots.  
    
    This method is not meant to be imported, the parameters are described in display_convolutions_per_filter 
    """
    output = np.array(output[:,:,:,filter_indices])
    if invert_cmap:
        cmap = 'gray_r'
    else:
        cmap = 'gray'
    if use_same_scaling:
        vmin = np.min(output)
        vmax = np.max(output)

    fig, ax= plt.subplots(len(images), len(filter_indices)+1, figsize=figsize, dpi=300)#sharex='col', sharey='row', 
    fig.tight_layout(h_pad= 2, w_pad= 1)
    for irow in range(len(images)):
        if images[irow].shape[2] == 1:
            plot = ax[irow,0].imshow(np.squeeze(images[irow]),cmap = 'gray')
        else:
            plot = ax[irow,0].imshow(images[irow])
        for icol in range(len(filter_indices)):   
            if use_same_scaling:
                plot = ax[irow,icol+1].imshow(output[irow,:,:,icol], vmin = vmin, vmax = vmax,cmap = cmap)
            else:
                plot = ax[irow,icol+1].imshow(output[irow,:,:,icol],cmap = cmap)

            ax[irow,icol+1].set_title('kernel '+str(filter_indices[icol]))
            if show_colorbar:
                divider = make_axes_locatable(ax[irow,icol+1])
                cax = divider.append_axes("right", size="5%", pad=0.08)
                fig.colorbar(plot, cax=cax)  

    if file_name != None:
        plt.savefig(file_name, dpi = dpi)
    

def display_convolutions_per_filter(model, conv_layer_name, input_images, figsize = (10,10), show_colorbar = True,
                         use_same_scaling = True, invert_cmap = False,filter_indices=None, dpi = None, file_name = None):
    """
    Display a set of convolutions for the images given.  

    

    Parameters
    ----------
    model : Model
        The keras model to be investigated
    conv_layer_name : String 
        the name of the layer, whose output should be displayed.  Can be retrieved with model.summary()
    input_images: [] 
        the input_image should have dimensions of either (n,x,y) for gray scale images of (n, x,y,3) with np.uint8 values
    show_colorbar: Boolean
        wether a colorbar should be displayed
    use_same_scaling: Boolean
        Set to true, it ensures the same scaling of the colorbar for all displayed convvolutions
    invert_cmap: Boolean
        inverts the display of the convolutions, high values are black now
    file_name: String
        if the file_name is not None the convolutions are saved to file, in which case figsize and dpi become important.
    figsize: Tuple(2)
        The size in inches of the saved figure.  Also influcences the output in a notebook
    dpi: int
        the dpi of the saved image
    filter_indices: []
        if None 4 kernels are randomly chosen, otherwise supply a list of kernel_indices

    Returns
    -------
    model
        the truncated model, with the layer specified as the last layer

    """
    layer, conv_layer_index = get_layer_by_name(model, conv_layer_name)
    if layer == None:
        print('No layer with name', conv_layer_name,'was found.')
        return
    output_shape = layer.output_shape
    if len(output_shape) != 4:
        print("Convolutions can only displayed for layers with a 4 dim output tensor.  The last dim is assumed to be \
               the number of channels")
        print("These have the following indices in the supplied model",get_conv_layer_indices(model))
        return
    if filter_indices == None:
        filter_indices = np.sort(np.random.choice(output_shape[3], size=4, replace=False))
    print('Dimensions of output from layer {} is {}.'.format(conv_layer_name,output_shape))
    print('{} images will be displayed together with the selected {} kernel indices'.format(len(input_images),len(filter_indices)))
#     trunc_model = create_trunc_model(model = model, index = conv_layer_index)
#     result = trunc_model.predict(x=input_images, batch_size=100, verbose=0)

    # generate the output at the desired level
    trunc_model = Model(inputs=model.input, outputs=model.get_layer(conv_layer_name).output)
    result = trunc_model.predict(input_images)
    
    # call the subroutinge which actually creates the plots
    show_images(input_images,result, filter_indices = filter_indices, use_same_scaling = use_same_scaling, 
                    invert_cmap = invert_cmap, figsize = figsize, dpi = dpi, show_colorbar = show_colorbar, file_name = file_name)
    return trunc_model
    