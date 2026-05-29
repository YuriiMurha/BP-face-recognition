# User Manual

This appendix describes how to install and operate the face recognition system on a personal computer. The system supports two modes of use: open-set recognition, in which new identities can be registered at runtime from a webcam, and closed-set recognition, in which the system classifies faces against a fixed set of identities known at training time. Both modes share the same detection front end and run on standard consumer hardware.

## A.2 Prerequisites

To install and run the system, the following components are required:

- Python 3.11 (3.11.x specifically; later versions will not work because TensorFlow 2.15 has no Python 3.12 wheel)
- The uv package manager
- Windows 10 or 11, or alternatively WSL2 with Ubuntu 22.04
- A webcam, either built-in or USB-connected
- For training on GPU: an NVIDIA card with at least 4 GB of VRAM and recent drivers installed on the Windows host

Training on CPU is supported but takes roughly three to four times longer than on a comparable GPU. The application itself, including detection and recognition, runs comfortably in real time on a modern CPU.

## A.3 Installation

The recommended installation path on Windows uses a project-local virtual environment created by uv.

```
git clone https://github.com/yourusername/BP-face-recognition.git
cd BP-face-recognition
uv venv .venv-win
.venv-win\Scripts\activate
uv sync
```

The uv sync step downloads every dependency listed in the lockfile and installs the correct wheel for the current platform. On a fresh machine the first sync takes several minutes; subsequent invocations are cached and complete in seconds.

For training under WSL2 the steps are analogous, but a separate virtual environment is created so that GPU and CPU TensorFlow wheels do not collide:

```
wsl --install -d Ubuntu-22.04
cd /mnt/d/Coding/Personal/BP-face-recognition
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv-wsl
source .venv-wsl/bin/activate
uv sync
```

GPU acceleration in WSL2 requires the host's NVIDIA driver to be visible from the Linux subsystem, which can be checked by running the nvidia-smi command from inside WSL. If GPU detection fails, the dedicated setup script automates the remaining configuration steps.

## A.4 Camera Configuration

The application reads camera settings from environment variables, with sensible defaults so that the typical user does not need to set anything. The defaults expect a webcam at device index zero, a resolution of 1280 by 720, and a frame rate of 30 frames per second.

To use a different webcam, set the camera device index before launching the application. To stream from a network camera, set the camera source type to RTSP and supply the stream URL. The full list of environment variables, with the role each plays, is given in the system manual.

If a smartphone is used as the webcam through a USB connection, most modern phones expose themselves as a standard USB video device when "USB camera" mode is enabled, and the application picks them up at one of the lower device indices, usually zero or one.

## A.5 Running the Application

The system is started through a single command that launches face detection, recognition, and the on-screen visualization in one process:

```
make run
```

On launch the application opens a window displaying the live camera feed with detected faces outlined in coloured rectangles and identity labels overlaid above each face. The colour of the rectangle indicates whether the face was matched to a known identity, was rejected as unknown, or could not be evaluated because the crop was too small. The window can be closed at any time by pressing the Q key or by closing it through the operating system.

To run in closed-set mode instead, in which the model directly classifies each face into one of fourteen known identities without consulting a database, the following command is used:

```
make run-closed-set
```

The closed-set mode is faster because no embedding similarity search is performed, but it cannot accept new identities at runtime. The choice between the two modes is purely operational; both share the same trained model and the same detection front end.

## A.6 Registering a New Identity

Open-set mode requires that at least one identity be present in the face database before the application is launched. New identities are added by running the registration script and supplying the name of the person to enrol:

```
make register name="Yurii"
```

The registration process opens the camera, asks the subject to position their face squarely in the frame, and captures ten samples spaced approximately one second apart. The samples are cropped, embedded by the FaceNet backbone, and written to the local face database. The subject is asked to slightly vary head pose between samples so that the resulting embeddings cover a small range of viewpoints rather than collapsing onto a single image.

The same person can be registered multiple times to accumulate additional samples; the database does not deduplicate names and treats each entry as an independent reference embedding. Removing an identity is done by editing the database file directly.

## A.7 Troubleshooting

A small number of installation problems recur often enough to be worth documenting.

If the application fails on startup with an error stating that no module named bp_face_recognition can be found, the most common cause is that the virtual environment was not activated, or that the project directory was not the working directory at launch time. The Makefile commands set the necessary environment variables automatically; running the Python scripts directly without Make requires that the source directory be on the Python path.

If dlib compilation fails on Windows, the project ships a pre-built dlib wheel for Python 3.11 that uv selects automatically; the failure typically indicates that the Python version has drifted away from the expected 3.11.x and is attempting to build dlib from source instead. Reinstalling with the correct Python version resolves the issue.

If TensorFlow reports that no GPU is detected under WSL2, the cause is almost always that the NVIDIA driver on the Windows host has not been updated to a version that supports CUDA in WSL. Updating the driver from the official NVIDIA page and rebooting the host is sufficient in nearly all cases.

If the camera window opens but shows a black frame, another application is most likely holding the device exclusively. Closing video conferencing applications and the system's privacy controls usually frees the device for use.

## A.8 Recovering Reference Materials

The complete source code of the system, together with the quantized (TensorFlow Lite) model weights, is stored on the CD attached to the thesis, with a README file at its root that mirrors the most-frequently-needed sections of this manual in a format suitable for reading without LaTeX rendering. The custom datasets and the full-precision training checkpoints are not on the CD: the datasets contain personal data, and the checkpoints can be regenerated by running the training pipeline; both are available from the author on request. The system manual that follows this user manual documents the internal organisation of the codebase for readers who wish to extend or modify the system.
