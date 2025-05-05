import napari
from magicgui import magicgui
from qtpy.QtWidgets import QFileDialog
import tifffile
import numpy as np
import json
from pathlib import Path


def load_first_timepoint(folder):
    folder = Path(folder)
    tiff_files = sorted(folder.glob("*.tif"))
    if not tiff_files:
        raise ValueError("No TIFF files found.")

    channel_files = {}
    for file in tiff_files:
        if "_ch0" in file.name:
            channel_files[0] = file
        elif "_ch1" in file.name:
            channel_files[1] = file
        # Add more channels as needed

    if 0 not in channel_files:
        raise ValueError("At least channel 0 must be present.")

    stack0 = tifffile.imread(channel_files[0])
    if 1 in channel_files:
        stack1 = tifffile.imread(channel_files[1])
        stack_rgb = np.stack([stack1, stack0, np.zeros_like(stack0)], axis=-1)
    else:
        stack_rgb = np.stack([stack0] * 3, axis=-1)

    return stack_rgb, folder


def generate_mips(volume):
    mip_xy = np.max(volume, axis=0)
    mip_yz = np.max(volume, axis=2).transpose(1, 0, 2)
    mip_xz = np.max(volume, axis=1).transpose(0, 1, 2)
    return mip_xy, mip_yz, mip_xz


@magicgui(call_button="Select folder and load", layout="vertical")
def folder_picker():
    folder = QFileDialog.getExistingDirectory(None, "Select image folder")
    if folder:
        volume, folder_path = load_first_timepoint(folder)
        mip_xy, mip_yz, mip_xz = generate_mips(volume)
        viewer.layers.clear()
        viewer.add_image(mip_xy, rgb=True, name="XY MIP")
        viewer.add_image(
            mip_yz, rgb=True, name="YZ MIP", translate=(0, volume.shape[0] + 20)
        )
        viewer.add_image(
            mip_xz, rgb=True, name="XZ MIP", translate=(volume.shape[1] + 20, 0)
        )
        viewer.dims.order = (1, 0)
        viewer.reset_view()
        folder_picker.folder = str(folder_path)


def export_config(event=None):
    if "XY MIP" not in viewer.layers:
        print("No image loaded.")
        return

    rect_layer = viewer.layers["XY Crop"] if "XY Crop" in viewer.layers else None
    line_layer = viewer.layers["Z Crop"] if "Z Crop" in viewer.layers else None

    if rect_layer is None or len(rect_layer.data) == 0:
        print("XY Crop not defined.")
        return

    if line_layer is None or len(line_layer.data) == 0:
        print("Z Crop not defined.")
        return

    rect = rect_layer.data[0]
    x_min, y_min = rect[0]
    x_max, y_max = rect[2]
    x_min, x_max = sorted([int(round(x_min)), int(round(x_max))])
    y_min, y_max = sorted([int(round(y_min)), int(round(y_max))])

    z_line = line_layer.data[0]
    z_start, z_end = sorted([int(round(z_line[0][1])), int(round(z_line[1][1]))])

    out = {
        "input_dir": folder_picker.folder,
        "output_dir": str(Path(folder_picker.folder).with_name("Cropped_Output")),
        "start_z": z_start,
        "end_z": z_end,
        "start_x": x_min,
        "end_x": x_max,
        "workers": 12,
    }

    out_path = Path(folder_picker.folder) / "config.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved config to", out_path)


viewer = napari.Viewer()
viewer.window.add_dock_widget(folder_picker, area="right")
viewer.bind_key("s", export_config, overwrite=True)


@viewer.bind_key("r")
def add_crop_layers(viewer):
    if "XY MIP" in viewer.layers and "XY Crop" not in viewer.layers:
        viewer.add_shapes(name="XY Crop", shape_type="rectangle", edge_color="yellow")
    if (
        "YZ MIP" in viewer.layers or "XZ MIP" in viewer.layers
    ) and "Z Crop" not in viewer.layers:
        viewer.add_shapes(name="Z Crop", shape_type="line", edge_color="cyan")


napari.run()
