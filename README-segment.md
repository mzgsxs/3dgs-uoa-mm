# Segmentation
Code to generate SAM2 and LTS segmentation results.
Though generated segmentation masks are not used in the final version, the code to generate masks is still provided.

## SAM2
Segment Anything 2 requires initial prompts, which is provided through `boxbounder.ipynb`.
The model is run on `mainsegremote.ipynb`.

### boxbounder.ipynb
A rough graphical solution to annotate images with bounding boxes.

The following pip packages are required:
- numpy
- matplotlib
- opencv-python

Be sure to tweak `inpath`, `outpath`, and `names` to appropriate directories.

It is sufficient to run all cells.

Note the following keybinds used to navigate the application:

menu keybinds:
| keybind | function |
| - | - |
| s or space | go into segment mode |
| c |   exit segment mode |
| w |  write, i.e. save (though this is done automatic) |
| f |  delete all bboxes |
| e |  continue to next |
| q |  go to last |
| g |  switch video by input |
| h |  switch cam by input |
| 1 |  stop |

### mainsegremote.ipynb
A headless version of segmentation suitable for barebones remote deployments.

This requires the following packages:
- numpy
- pytorch
- imageio
- opencv-python-headless

This will also require the SAM 2 package and the `sam2.1_hiera_large.pt` model file, which are available from https://github.com/facebookresearch/sam2.

Tweak `sam2_checkpoint` to the correct directory for the model.

Be sure to tweak `inpath`, `boxpath`, `outpath`, and `names` to appropriate directories.