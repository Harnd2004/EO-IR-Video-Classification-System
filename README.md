# EO/IR Video Classification System
## Project Overview
An advanced, real-time AI surveillance system developed to identify military entities in EO/IR video feeds. This system goes beyond simple detection by providing automated **Threat Assessment** and generating comprehensive **Intelligence Reports** in multiple formats (JSON, PDF, and DOCX).



##  Key Features
* **YOLOv8 Core:** High-speed detection of 7 military categories: Artillery, Missiles, Radar, Rocket Launchers, Soldiers, Tanks, and Vehicles.
* **Automated Threat Assessment:** Real-time classification of detections into Critical, High, Medium, or Low threat levels based on entity type.
* **Intelligence Reporting Engine:**
    * **Interactive HTML Dashboard:** Visual summaries of detection patterns and confidence metrics.
    * **Multi-Format Export:** One-click generation of professional PDF, Word, and JSON reports for command-level review.
* **Side-by-Side Analysis:** Dual-video interface to compare raw surveillance footage against AI-annotated streams.
* **Edge-Ready Logic:** Optimized processing pipeline designed for rapid inference.

##  Technical Stack
* **Model:** YOLOv8 (Ultralytics)
* **Interface:** Gradio
* **Vision:** OpenCV, MediaPipe
* **Backend:** Python, Pandas, NumPy
* **Document Generation:** FPDF2 (PDF), Python-Docx (Word)

##  Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Harnd2004/EO-IR-Video-Classification-System.git
   cd EO-IR-Video-Classification-System
2. **Install dependencies:**
     ```bash
   pip install ultralytics gradio pandas opencv-python fpdf2 python-docx
3. **Run the application:**
    ```bash
   python app.py
    
##  Intelligence Report Sample

* The system generates structured reports including:

* Executive Summary: Total detection counts and average confidence.

* Threat Assessment: Breakdown of critical vs. routine concerns.

* Detection Log: Precise timestamps, durations, and spatial coordinates for every entity found.

##  UI
![WhatsApp Image 2026-02-20 at 5 55 59 PM](https://github.com/user-attachments/assets/ff73b815-d628-4042-ac91-e7a187c69ac7)
![WhatsApp Image 2026-02-20 at 5 55 59 PM (1)](https://github.com/user-attachments/assets/27623eb1-030e-4c45-aa41-3e822734f7bc)
![WhatsApp Image 2026-02-20 at 5 55 58 PM](https://github.com/user-attachments/assets/90b2afe1-47bd-4c3d-a856-9c6b2d930424)
![WhatsApp Image 2026-02-20 at 5 56 00 PM](https://github.com/user-attachments/assets/8f04d784-1764-47aa-b717-ce111b329b07)

