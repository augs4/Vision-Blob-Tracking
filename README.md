# Vision Blob Tracking

A real-time computer vision tool for detecting, tracking, and visualizing blobs in video streams or images.  
Built with **Streamlit**, **OpenCV**, and **NumPy**, it provides interactive controls for analysis, customization, and export.

## 🚀 Features
- **Real-time Blob Detection** using OpenCV
- **ROI Selection & Inversion** – focus only on areas you care about
- **Customizable Colors** for points, boxes, and labels
- **Grid Overlays** with adjustable divisions
- **Curved Motion Links** to visualize movement paths
- **Zoom & Scale Controls**
- **MP4 Export** of processed results

## 📦 Installation (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy to Streamlit Community Cloud
1. Push this folder to a **public** GitHub repository.
2. Go to https://share.streamlit.io and click **New app**.
3. Select your repo and branch, set **Main file path** to `app.py`, then **Deploy**.

## 🗂 Repo structure
```
.
├─ app.py               # Streamlit entry point (copied from your uploaded script)
├─ requirements.txt     # Dependencies for Streamlit Cloud
├─ .gitignore           # Keeps envs/outputs out of git
└─ README.md
```

## 🛠 Requirements
- Python 3.9–3.11
- Streamlit, OpenCV, NumPy (see `requirements.txt`)

## 📄 License
MIT (recommend adding a LICENSE file if you plan to share/accept contributions)