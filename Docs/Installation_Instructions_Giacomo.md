# CAS AI for Creative Practices
## Installation Guide for Input Session by Giacomo Lepri
This guide helps you download the input-session materials, install Pure Data (Pd), and install the `nn~` object used to run RAVE models in Pd.

> **Important:** 
>
> - Complete the sections in the order shown.
> - Follow the verification or test step after each installation before continuing.
> - Download the version of each program that matches your operating system.
> - For Pure Data and nn~, download precompiled application or package files. Do not download source code.
> - These instructions cover macOS and Windows.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Download the course material](#2-download-the-course-material)
3. [Install Pure Data](#3-install-pure-data)
4. [Install the nn~ object](#4-install-the-nn-object)
5. [Test the installation](#5-test-the-installation)
6. [Troubleshooting checklist](#6-troubleshooting-checklist)

---

## 1. Before you start

You will need:

- A computer running macOS or Windows.
- A stable internet connection.
- Enough free disk space for the session materials and software.
- Permission to install applications. You may be asked for an administrator password or approval.

### Terms used in this guide

| Term | Meaning |
|---|---|
| **Pure Data (Pd)** | A visual programming environment for audio, music, and multimedia. |
| **Patch** | A Pure Data project file. Patch files use the `.pd` extension. |
| **Object** | A reusable component that adds a function to Pure Data. The `nn~` object lets Pd use neural-network audio models. |
| **DSP** | Digital Signal Processing. In Pd, DSP must be on before audio can be heard. |
| **RAVE** | A neural audio model that can be used for real-time audio processing and generation. |

---

## 2. Download the course material

The session materials are provided in a Google Drive folder.

1. Open [the course-material folder in Google Drive](https://drive.google.com/drive/folders/14xMmuIaipBPtCaAKzbML0UnByqoHOGwG).
2. Download the course material.
4. If the material downloads as a ZIP file, double-click the ZIP file to unpack it.

### Check your work

Open the downloaded course-material folder. It should contain the session files, including Pure Data patches or example files such as:

```text
03MAIN_water.pd
```

---

## 3. Install Pure Data

Pure Data (Pd) is the application used to open and run the course patches.

### Download Pure Data

1. Open the [official Pure Data download page](https://puredata.info/downloads/pure-data).
2. Choose a **precompiled** download for your operating system:

| Operating system | Download to choose |
|---|---|
| macOS | The macOS disk-image download. |
| Windows | The Windows installer. |

### macOS: install Pure Data

1. Find the downloaded file. It may have a name similar to:

   ```text
   Pd-0.56-1.dmg
   ```

2. Double-click the `.dmg` file.
3. A Finder window opens. Drag the Pd application to the **Applications** folder.
4. When copying finishes, open the Applications folder.
5. Start Pd by double-clicking the Pd application.

If macOS blocks the application:

1. Confirm that you downloaded Pd from the official Pure Data download page.
2. Control-click the Pd application and choose **Open**.
3. Confirm **Open** in the security dialog, if shown.

### Windows: install Pure Data

1. Find the downloaded installer. It may have a name similar to:

   ```text
   pd-0.56-1.windows-installer.exe
   ```

2. Double-click the installer.
3. If Windows displays a security warning, first confirm that you downloaded the installer from the official Pure Data download page. Then choose **Run anyway**, if that option is available.
4. Continue through the installation setup using the default options.
5. Finish the installation.
6. Start Pd from the **Start** menu.

### First start and verification

1. Start Pure Data.
2. Grant Pd the permissions it requests, if any. This can include permission to access files, the microphone, or audio devices.
3. Close Pd after it has opened successfully.

The first start creates Pd's user folder inside your **Documents** folder. The `nn~` installation may need this folder in the next section.

**Success:** Pd opens and closes without an error message.

---

## 4. Install the nn~ object

The `nn~` object allows Pure Data patches to load and use neural-network models, including RAVE models.

### Download nn~

1. Open the [nn~ version 1.6.0 release page](https://github.com/acids-ircam/nn_tilde/releases/tag/v1.6.0).
2. Scroll to the bottom of the release page and find the downloadable files in the **Assets** section.
3. Download the **Pure Data** version for your operating system. Do **not** download a Max MSP version.

| Operating system | Package to download |
|---|---|
| macOS | `nn_puredata_macos_ub` |
| Windows | `nn_puredata_window_x64` |

### Install nn~

The official installation instructions are available in [the nn~ repository documentation](https://github.com/acids-ircam/nn_tilde?tab=readme-ov-file).

Follow the installation instructions for **Pure Data**, not the instructions for Max MSP.

1. If the downloaded `nn~` file is a ZIP archive, double-click it to unpack it.
2. Follow the Pure Data installation instructions in the official `nn~` documentation for your operating system.
3. Install the files into the Pd location specified in that documentation.
5. When installation is complete, quit Pd completely if it is still open.

---

## 5. Test the installation

Use one of the downloaded session patches to confirm that Pd, `nn~`, and audio output work together.

1. Make sure Pure Data is completely closed.
2. Open the downloaded course-material folder.
3. Double-click an example Pure Data patch, for example:

   ```text
   03MAIN_water.pd
   ```

4. Pd starts and opens the patch.
5. In the main Pd window, turn on the **DSP** toggle.
   - It is usually labelled `DSP`.
   - It must be enabled before Pd can produce sound.
6. In the opened Pd patch, press the **Start** button.
7. Listen through your selected speakers or headphones.

### Success criteria

- No `nn~` or other error messages appear in the Pd console.
- You hear the expected screaming sound from the example patch.

If both conditions are true, your computer is ready for the input session by Giacomo Lepri.

---

