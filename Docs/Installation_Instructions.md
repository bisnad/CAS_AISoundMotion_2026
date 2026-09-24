# CAS AI for Creative Practices
## Installation Guide for Module 3: AI for Sound and Module 5: AI for Motion

This guide prepares your computer for the course examples. It is written for students who may be new to command-line tools.

> **Important:** Complete the sections in order. After each installation, run the verification command before continuing. You need a reliable internet connection and may be asked for administrator permission.

> **macOS support:** Only Apple computers with Apple silicon are supported, for example Macs with M1, M2, M3, or later Apple chips. **Intel-based Macs are not supported** by this course installation.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Install a compilation environment](#2-install-a-compilation-environment)
3. [Install Git and Git LFS](#3-install-git-and-git-lfs)
4. [Install uv](#4-install-uv)
5. [Download the course repositories](#5-download-the-course-repositories)
6. [Install the course Python environments](#6-install-the-course-python-environments)
7. [Test course examples](#7-test-course-examples)
8. [Set up Visual Studio Code trust](#8-set-up-visual-studio-code-trust)
9. [Gain access to Hugging Face models](#9-gain-access-to-hugging-face-models)
10. [Troubleshooting checklist](#10-troubleshooting-checklist)

---

## 1. Before you start

### Supported computers

| Operating system | Support status |
|---|---|
| **Windows** | Supported. Select the appropriate course installation script depending on whether the computer has an NVIDIA GPU. |
| **macOS** | Supported only on Macs with Apple silicon, such as M1, M2, M3, or later chips. Intel-based Macs are not supported. |

### Check whether your Mac is supported

1. Click the Apple menu in the top-left corner of the screen.
2. Select **About This Mac**.
3. Look for **Chip**. It should say `Apple M1`, `Apple M2`, `Apple M3`, or a later Apple chip.
4. If it says **Processor** followed by `Intel`, your Mac is not supported by this installation guide.

You will install these three tools before downloading the course material:

| Tool | Why you need it |
|---|---|
| **Compilation environment** | Lets Python install components that must be built on your computer. On macOS, it also installs Git. |
| **Git** | Downloads and updates the course files. |
| **uv** | Creates and runs the Python environments used by the examples. |

### Terms used in this guide

- **PowerShell**: Windows' command-line application.
- **Terminal**: macOS's command-line application.
- **Command**: a line of text you paste or type into PowerShell or Terminal, then press Enter.
- **Repository**: a course folder downloaded from GitHub.
- **Script**: a file that starts an installation or course example. Windows scripts end in `.bat`; macOS scripts end in `.sh`.

> **Copy commands exactly.** The punctuation in a command matters.

---

## 2. Install a compilation environment

Some Python packages used in the course need compiler tools during installation. On macOS, this step also installs Git.

### Windows: install Visual Studio Build Tools

1. Open **PowerShell**.
2. Copy the complete command below into PowerShell and press Enter. This is one command; the backtick character (`` ` ``) continues it onto the next line.

```powershell
winget install --exact --id Microsoft.VisualStudio.2022.BuildTools `
  --override "--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

3. If Windows shows a UAC permission window, choose **Yes**.
4. Wait for the installation to finish. This can take some time.
5. Close every currently open PowerShell window.

#### Verify the compiler

1. Open the **Start** menu.
2. Search for and open:

```text
x64 Native Tools Command Prompt for VS 2022
```

3. Run:

```text
cl
```

**Success:** The output begins with text similar to:

```text
Microsoft (R) C/C++ Optimizing Compiler Version ...
```

### macOS with Apple silicon only: install Xcode Command Line Tools

This section applies only to supported Macs with Apple silicon. Intel Macs are not supported.

1. Open **Terminal**.
2. Run:

```bash
xcode-select --install
```

3. A macOS dialog appears. Choose **Install**.
4. Accept any licence or permission requests and wait until installation completes.
5. Close Terminal, then open a new Terminal window.

#### Verify the compiler and Git

Run these commands one at a time:

```bash
clang --version
make --version
git --version
```

**Success:** Each command displays version information rather than an error.

---

## 3. Install Git and Git LFS

Git downloads the course repositories. Git LFS (*Large File Storage*) downloads large files, such as audio, video, motion-capture data, and model weights.

### Windows: install Git

1. Open the **Start** menu.
2. Search for **PowerShell** and open it.
3. Paste this command and press Enter:

```powershell
winget install --exact --id Git.Git --source winget
```

4. If Windows displays a **User Account Control (UAC)** prompt, choose **Yes**.
5. When the installation is complete, close PowerShell.

#### Verify Git

Open a **new** PowerShell window and run:

```powershell
git --version
```

**Success:** You should see a version number, such as `git version 2.x.x`.

### Windows: install Git LFS

1. Open [git-lfs.com](https://git-lfs.com/) in a web browser.
2. Download and install the Windows version of Git LFS.
3. Open a new PowerShell window.
4. Run:

```powershell
git lfs install
```

#### Verify Git LFS

```powershell
git lfs version
```

**Success:** You should see a Git LFS version number.

### macOS with Apple silicon only: install Git LFS

Git was installed in Section 2 with the Xcode Command Line Tools. This section installs Git LFS.

1. Open [git-lfs.com](https://git-lfs.com/) in a web browser.
2. Download the **Apple Silicon** version of Git LFS for macOS.
3. Open the downloaded ZIP file to unpack it.
4. Open **Terminal**.
5. Run:

```bash
cd ~/Downloads/git-lfs-*
sudo ./install.sh
```

6. Enter your Mac password if requested. Nothing appears while you type the password; this is normal. Press Enter when finished.

#### Verify Git LFS

Open a new Terminal window and run:

```bash
git lfs version
```

**Success:** You should see a Git LFS version number.

---

## 4. Install uv

[`uv`](https://docs.astral.sh/uv/) manages the Python environments and packages used by the course examples.

### Windows: install uv

1. Open **PowerShell**.
2. Run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. When installation finishes, close PowerShell.
4. Open a **new** PowerShell window.

#### Verify uv

```powershell
uv --version
```

**Success:** You should see a uv version number.

### macOS with Apple silicon only: install uv

1. Open **Terminal**.
2. Run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. When the installation finishes, close Terminal.
4. Open a **new** Terminal window.

#### Verify uv

```bash
uv --version
```

**Success:** You should see a uv version number.

> If Terminal says `uv: command not found`, close Terminal completely, open it again, and retry. If it still fails, contact the course team.

---

## 5. Download the course repositories

There are two course repositories:

| Repository | Status | Contents |
|---|---|---|
| `CAS_AISoundMotion_2026` | **Required** | Source code, examples, and installation scripts. |
| `CAS_AISoundMotion_Data_2026` | **Optional** | Additional course data, including video, audio, motion-capture files, and model weights. |

You **must** download `CAS_AISoundMotion_2026`. Download `CAS_AISoundMotion_Data_2026` only when you are instructed to do so for a particular class activity or example.

> The optional data repository can be large. Downloading it requires a stable internet connection and sufficient free disk space.

### Create a course folder

1. Open **File Explorer** on Windows or **Finder** on macOS.
2. Choose an easy-to-find location, for example **Documents**.
3. Create a new folder, for example:

```text
CAS_AI_Course
```

4. Open the folder and copy its path.

Helpful ways to get a folder path:

- **Windows:** Click the File Explorer address bar and copy the displayed path.
- **macOS:** Hold Option while right-clicking the folder and choose **Copy ... as Pathname** if available. Alternatively, type `cd ` in Terminal and drag the folder into the Terminal window.

### Clone the required course repository

1. Open **PowerShell** on Windows or **Terminal** on macOS.
2. Move into your course folder. Replace `<path-to-your-course-folder>` with the folder path you created.

```bash
cd <path-to-your-course-folder>
```

Examples:

```powershell
# Windows
cd C:\Users\YourName\Documents\CAS_AI_Course
```

```bash
# macOS
cd ~/Documents/CAS_AI_Course
```

3. Download the required course code repository:

```bash
git clone https://github.com/bisnad/CAS_AISoundMotion_2026.git
```

4. Wait for the command to complete. Do not close the window during the download.

#### Check your work

Your course folder should now contain:

```text
CAS_AISoundMotion_2026
```

### Optional: clone the course data repository

Download this repository only if you have been instructed to use the additional data files.

1. Make sure PowerShell or Terminal is still in your course folder. If necessary, run:

```bash
cd <path-to-your-course-folder>
```

2. Download the optional data repository:

```bash
git clone https://github.com/bisnad/CAS_AISoundMotion_Data_2026.git
```

3. Wait for the command to complete. Do not close the window during the download.

#### Check your work

Your course folder should now also contain:

```text
CAS_AISoundMotion_Data_2026
```

> If the download skips large files or shows Git LFS errors, repeat the Git LFS installation in [Section 3](#3-install-git-and-git-lfs).

### macOS only: open `.sh` files in Terminal

Configure Finder to open course shell scripts in Terminal:

1. In Finder, open `CAS_AISoundMotion_2026`.
2. Navigate to `Installers/run_installation_macos-cpu.sh`.
3. Select the file and press **Command + I** to open **Get Info**.
4. Under **Open with**, select **Terminal**.
5. Click **Change All...** if you want every `.sh` file to open in Terminal.
6. Confirm the change if macOS asks.

---

## 6. Install the course Python environments

The installation script prepares each example folder for your operating system. It copies the correct `uv` project files (`pyproject.toml`) to the relevant folders.

### macOS with Apple silicon: run the installation script

1. In Finder, open:

```text
CAS_AISoundMotion_2026/Installers
```

2. Double-click:

```text
run_installation_macos-cpu.sh
```

3. Terminal opens and runs the script.
4. Wait until the script completes, and read the final message in Terminal.

### Windows: run the installation script

First determine whether your PC has an NVIDIA graphics card (GPU):

1. Right-click the **Start** button and choose **Device Manager**.
2. Expand **Display adapters**.
3. If an item includes **NVIDIA**, use the NVIDIA installer. Otherwise, use the CPU installer.

| Your computer | Script to run |
|---|---|
| Windows PC with an NVIDIA GPU | `CAS_AISoundMotion_2026/Installers/run_installation_windows-cu126.bat` |
| Windows PC without an NVIDIA GPU | `CAS_AISoundMotion_2026/Installers/run_installation_windows-cpu.bat` |

Then:

1. Open File Explorer.
2. Navigate to `CAS_AISoundMotion_2026/Installers`.
3. Double-click the correct `.bat` file.
4. A command window opens. Wait for it to finish.

> If Windows displays a security warning, first confirm that the file is inside the `CAS_AISoundMotion_2026` folder downloaded from the official course link. Then choose to run it.

---

## 7. Test course examples

Each example has a launcher script:

- **macOS:** `.sh` file
- **Windows:** `.bat` file

When you run an example for the first time, `uv` automatically creates a Python environment and downloads the required packages. This can take a long time, particularly for examples that use machine-learning libraries or models. Later runs should be much faster.

For now, only confirm that applications start. Do not wait for long training processes to finish. To stop a training example, close the Terminal or command window it opened.

> Some examples may require files from the optional `CAS_AISoundMotion_Data_2026` repository. If an example reports missing data, download that repository as described in [Section 5](#5-download-the-course-repositories).

> Examples that use gated Hugging Face models require the setup in [Section 9](#9-gain-access-to-hugging-face-models).

### Module 3: AI for Sound

All paths below are relative to the `CAS_AISoundMotion_2026` repository folder.

#### macOS launchers

```text
AudioAnalysis/audio_analysis.sh
AudioAutoencoder/Inference/HelloWorld/vae_vocos_cnn_hello_world.sh
AudioAutoencoder/Inference/LatentManipulation/vae_vocos_cnn_interactive.sh
AudioAutoencoder/Inference/LatentNavigation/vae_vocos_cnn_interactive.sh
AudioAutoencoder/Training/audio_vae_vocos_cnn.sh
AudioClustering/audio_clustering.sh
AudioNearestNeighbors/audio_nearest_neighbors.sh
AudioRepresentation/audio_representation_hello_world.sh
HuggingFace/audio_ldm2_hello_world.sh
HuggingFace/stable_audio_open1_hello_world.sh
SpectralPlayground/spectral_playground_hello_world.sh
StableAudio3/stable_audio_3_gui_inpaint_continue.sh
StableAudio3/stable_audio_3_gui_interpolation.sh
StableAudio3/stable_audio_3_gui_text_to_audio.sh
StableAudio3/stable_audio_3_hello_world.sh
```

#### Windows launchers

```text
AudioAnalysis/audio_analysis.bat
AudioAutoencoder/Inference/HelloWorld/vae_vocos_cnn_hello_world.bat
AudioAutoencoder/Inference/LatentManipulation/vae_vocos_cnn_interactive.bat
AudioAutoencoder/Inference/LatentNavigation/vae_vocos_cnn_interactive.bat
AudioAutoencoder/Training/audio_vae_vocos_cnn.bat
AudioClustering/audio_clustering.bat
AudioNearestNeighbors/audio_nearest_neighbors.bat
AudioRepresentation/audio_representation_hello_world.bat
HuggingFace/audio_ldm2_hello_world.bat
HuggingFace/stable_audio_open1_hello_world.bat
SpectralPlayground/spectral_playground_hello_world.bat
StableAudio3/stable_audio_3_gui_inpaint_continue.bat
StableAudio3/stable_audio_3_gui_interpolation.bat
StableAudio3/stable_audio_3_gui_text_to_audio.bat
StableAudio3/stable_audio_3_hello_world.bat
```

### Module 5: AI for Motion

All paths below are relative to the `CAS_AISoundMotion_2026` repository folder.

#### macOS launchers

```text
MocapClassifier/Inference/mocap_classifier_interactive_lstm.sh
MocapClassifier/Training/mocap_classifier_lstm.sh
MocapClustering/clustering_interactive.sh
MocapPlayer/mocap_player.sh
MotionAnalysis/mocap_analysis.sh
MotionAutoencoder/Inference/LatentManipulation/vae_cnn_interactive.sh
MotionAutoencoder/Inference/LatentNavigation/vae_cnn_interactive.sh
MotionAutoencoder/Training/vae_cnn.sh
MotionContinuation/Inference/td_interactive.sh
MotionContinuation/Training/td.sh
MotionTracking/Mediapipe/PoseEstimation/mediapipe_pose.sh
MotionTracking/Yolo/ObjectDetection/yolo_object2d.sh
MotionTracking/Yolo/PoseEstimation/yolo_pose3d.sh
SensorClassifier/Inference/sensor_classifier_interactive.sh
SensorClassifier/Training/sensor_classifier.sh
SensorPlayer/sensor_player.sh
SensorRecorder/sensor_recorder.sh
```

#### Windows launchers

```text
MocapClassifier/Inference/mocap_classifier_interactive_lstm.bat
MocapClassifier/Training/mocap_classifier_lstm.bat
MocapClustering/clustering_interactive.bat
MocapPlayer/mocap_player.bat
MotionAnalysis/mocap_analysis.bat
MotionAutoencoder/Inference/LatentManipulation/vae_cnn_interactive.bat
MotionAutoencoder/Inference/LatentNavigation/vae_cnn_interactive.bat
MotionAutoencoder/Training/vae_cnn.bat
MotionContinuation/Inference/td_interactive.bat
MotionContinuation/Training/td.bat
MotionTracking/Mediapipe/PoseEstimation/mediapipe_pose.bat
MotionTracking/Yolo/ObjectDetection/yolo_object2d.bat
MotionTracking/Yolo/PoseEstimation/yolo_pose3d.bat
SensorClassifier/Inference/sensor_classifier_interactive.bat
SensorClassifier/Training/sensor_classifier.bat
SensorPlayer/sensor_player.bat
SensorRecorder/sensor_recorder.bat
```

---

## 8. Set up Visual Studio Code trust

Visual Studio Code may ask whether you trust the course folders. Trust is required before the editor can run certain project features and scripts.

1. Open **Visual Studio Code**.
2. Open one of the example folders from the course repository.
3. Open the Command Palette:
   - **macOS:** Command + Shift + P
   - **Windows:** Ctrl + Shift + P
4. Type:

```text
Workspaces: Manage Workspace Trust
```

5. Select the command from the list.
6. Choose **Trust**.
7. If Visual Studio Code asks whether to trust all folders in the downloaded GitHub repository, confirm only if you downloaded the repository from the official course link in this guide.

---

## 9. Gain access to Hugging Face models

Some examples download models from [Hugging Face](https://huggingface.co/). Several of these models are **gated**: before your computer can download them, you must create an account, accept the model conditions, and log in from your computer.

### Create a Hugging Face account

1. Open [huggingface.co](https://huggingface.co/).
2. Select **Sign Up**.
3. Create an account with your email address and a username.
4. Verify your email address.
5. Sign in in the web browser.

### Request access to Stable Audio Open 1.0

1. While signed in, open [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0).
2. Read the displayed licence and access conditions.
3. Complete any required fields.
4. Select the access button. Depending on the page, it may say:

```text
Agree and access repository
Request access
Accept and continue
```

### Request access to Stable Audio 3 Small-SFX

1. While signed in, open [Stable Audio 3 Small-SFX](https://huggingface.co/stabilityai/stable-audio-3-small-sfx).
2. Read the displayed licence and access conditions.
3. Complete any required fields.
4. Select the access button. It may say:

```text
Agree and access repository
Request access
Accept and continue
```

### Create a read token

A token is a private credential that allows software on your computer to access models using your Hugging Face account.

1. Open [Hugging Face token settings](https://huggingface.co/settings/tokens).
2. Click **New token**.
3. Give it a meaningful name, for example:

```text
CAS AI course laptop
```

4. Choose a token with **read-only** access.
5. Click **Generate token**.
6. Copy the token and store it somewhere safe.

> **Keep your token secret.** Treat it like a password. Never share it in chat, email, screenshots, or course submissions. Do not put it in GitHub or another public repository.

### Log in from your computer

1. Open Terminal on macOS or PowerShell on Windows.
2. Move into this folder. Replace `<path-to-your-course-folder>` with the actual path to your course folder:

```bash
cd <path-to-your-course-folder>/CAS_AISoundMotion_2026/AI_Sound
```

3. Start Hugging Face login:

```bash
uv run hf auth login
```

4. When prompted, paste your personal read token and press Enter.
5. Check that login succeeded:

```bash
uv run hf auth whoami
```

**Success:** The command displays your Hugging Face username.

---

## 10. Troubleshooting checklist

Before contacting the course team, check the following:

1. On macOS, confirm that your computer has Apple silicon. Intel Macs are not supported.
2. Open a **new** PowerShell or Terminal window after installing compiler tools, Git, Git LFS, or `uv`.
3. Confirm that the relevant verification command displays a version number.
4. Confirm that the required repository folder exists:

   ```text
   CAS_AISoundMotion_2026
   ```

5. Download `CAS_AISoundMotion_Data_2026` only when it is needed. If an example reports missing course data, download it using [Section 5](#5-download-the-course-repositories).
6. If large optional course files are missing, confirm that Git LFS is installed.
7. On Windows, choose the correct installer:
   - `cu126` for a PC with an NVIDIA GPU.
   - `cpu` for a PC without an NVIDIA GPU.
8. Allow additional time when launching an example for the first time: `uv` may need to download and install many packages.
9. Complete the Hugging Face access and login steps before launching examples that require Stable Audio models.
10. When reporting a problem, include:
   - Your operating system and version.
   - Your Mac chip, if you are using macOS.
   - The exact command or script you ran.
   - A screenshot or copied text of the complete error message.
   - Whether your Windows PC has an NVIDIA GPU.

---

**End of installation guide**
