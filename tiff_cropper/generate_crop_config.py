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

    # Squeeze to remove singleton dimensions (like T or C)
    stack0 = tifffile.imread(channel_files[0]).squeeze()

    if 1 in channel_files:
        stack1 = tifffile.imread(channel_files[1]).squeeze()

        # Ensure we have 3D arrays (Z, Y, X)
        if stack0.ndim != 3 or stack1.ndim != 3:
            raise ValueError(
                "Input TIFs must be 3D (or squeezable to 3D) after loading. "
                f"Got shapes: Ch0 {stack0.shape}, Ch1 {stack1.shape}"
            )

        stack_rgb = np.stack([stack1, stack0, np.zeros_like(stack0)], axis=-1)
    else:
        # Ensure we have a 3D array (Z, Y, X)
        if stack0.ndim != 3:
            raise ValueError(
                "Input TIF must be 3D (or squeezable to 3D) after loading. "
                f"Got shape: Ch0 {stack0.shape}"
            )
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
    # Note: Napari 0.4.18 changed rectangle coordinates. This assumes (y, x) indexing.
    # rect[0] = [y_min, x_min]
    # rect[2] = [y_max, x_max]
    y_min, x_min = rect[0]
    y_max, x_max = rect[2]

    x_min, x_max = sorted([int(round(x_min)), int(round(x_max))])
    y_min, y_max = sorted([int(round(y_min)), int(round(y_max))])

    z_line = line_layer.data[0]
    # Napari line shape is [[z_start, y_or_x_start], [z_end, y_or_x_end]]
    # We only care about the z-axis, which is axis 0 for YZ/XZ MIPs in this layout
    # Wait, the add_image for YZ/XZ uses default (0, 1) dims.
    # YZ MIP: (Y, Z, C). Translate(0, Z+20). Dims (1,0) -> (Z, Y)
    # XZ MIP: (Z, X, C). Translate(Y+20, 0). Dims (1,0) -> (X, Z)
    # The crop layers are added to these.
    # Let's assume the Z Crop line is drawn on the XZ MIP (shape (Z, X)).
    # The line coordinates will be [[z_start, x_start], [z_end, x_end]]
    # We should be indexing the Z-axis, which is axis 0.

    # Re-reading your code:
    # YZ MIP translate=(0, volume.shape[0] + 20) -> (0, Z+20). volume.shape[0] is Z.
    # XZ MIP translate=(volume.shape[1] + 20, 0) -> (Y+20, 0). volume.shape[1] is Y.
    # viewer.dims.order = (1, 0)

    # This means:
    # YZ MIP (shape Y, Z, C) is displayed with dims (Z, Y).
    # XZ MIP (shape Z, X, C) is displayed with dims (X, Z).

    # If Z-line is drawn on XZ MIP (dims X, Z), coords are [x, z]. We want z.
    # z_start = z_line[0][1]
    # z_end = z_line[1][1]

    # If Z-line is drawn on YZ MIP (dims Z, Y), coords are [z, y]. We want z.
    # z_start = z_line[0][0]
    # z_end = z_line[1][0]

    # Your original code used z_line[0][1] and z_line[1][1]. This implies you
    # intended the line to be drawn on the XZ MIP.

    z_start, z_end = sorted([int(round(z_line[0][1])), int(round(z_line[1][1]))])

    out = {
        "input_dir": folder_picker.folder,
        "output_dir": str(Path(folder_picker.folder).with_name("Cropped_Output")),
        "start_z": z_start,
        "end_z": z_end,
        "start_x": x_min,
        "end_x": x_max,
        # The napari tool doesn't define Y crop, but crop_tool.py doesn't use it anyway
        "start_y": y_min,
        "end_y": y_max,
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
        # Get shape from the layer to set bounds
        layer_shape = viewer.layers["XY MIP"].data.shape
        # Default rectangle (y, x)
        default_rect = [
            [layer_shape[0] * 0.1, layer_shape[1] * 0.1],  # top-left
            [layer_shape[0] * 0.1, layer_shape[1] * 0.9],  # top-right
            [layer_shape[0] * 0.9, layer_shape[1] * 0.9],  # bottom-right
            [layer_shape[0] * 0.9, layer_shape[1] * 0.1],  # bottom-left
        ]
        viewer.add_shapes(
            default_rect,
            name="XY Crop",
            shape_type="rectangle",
            edge_color="yellow",
            face_color="transparent",
            edge_width=2,
        )
    if "XZ MIP" in viewer.layers and "Z Crop" not in viewer.layers:
        # Use XZ MIP for Z cropping
        layer_shape = viewer.layers["XZ MIP"].data.shape  # (Z, X, C)
        # Dims are (X, Z)
        # z_dim_index = 1 # Z is the 2nd dim displayed
        # x_dim_index = 0 # X is the 1st dim displayed

        # Default line [[x1, z1], [x2, z2]]
        default_line = [
            [layer_shape[1] * 0.1, layer_shape[0] * 0.1],  # (x, z) start
            [layer_shape[1] * 0.9, layer_shape[0] * 0.9],  # (x, z) end
        ]
        viewer.add_shapes(
            default_line,
            name="Z Crop",
            shape_type="line",
            edge_color="cyan",
            edge_width=2,
        )


napari.run()
