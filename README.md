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
   git clone [https://github.com/Harnd2004/EO-IR-Video-Classification-System.git](https://github.com/Harnd2004/EO-IR-Video-Classification-System.git)
   cd EO-IR-Video-Classification-System
