### **Project Details**

Our project is a **multi-dataset fine-tuned exoplanet classification model** that leverages data from NASA’s **Kepler (KOI)**, **K2**, and **TESS** missions. The goal was to build a robust, accurate, and efficient model capable of identifying exoplanet candidates across varying data domains — something traditional models often struggle with due to dataset shifts and observational differences.

We used a **Random Forest** as the core model architecture, fine-tuned in a multi-stage process:

1. Initial training on KOI.
2. Fine-tuning on K2.
3. Final tuning on TESS, excluding validation data to prevent leakage.

To improve efficiency, we used **Random Forest Feature Importance** for **feature selection**, reducing dimensionality while boosting performance. Loss functions were adapted dynamically during each stage to prevent overfitting and encourage generalization across datasets. Our final model achieved **93% accuracy** on a combined validation set, outperforming benchmarks from published ensemble models (~84%).

We also developed a **Flask/FastAPI-based backend** for real-time inference. This API supports **Model-as-a-Service architecture**, allowing rapid deployment and smooth data integration for future NASA missions or researchers.

---

### **Tools & Technologies Used**

* **Programming Language:** Python
* **Libraries & Frameworks:** scikit-learn (Random Forest), pandas, NumPy, Flask, FastAPI
* **Data:** NASA KOI, K2, and TESS exoplanet datasets
* **Deployment:** Local inference server with modular APIs

---

### **Benefits & Impact**

* **Higher Accuracy:** 93% on validation — a ~10% improvement over published baselines.
* **Cross-Mission Reliability:** Fine-tuned across three missions, ensuring domain-agnostic performance.
* **Scientific Value:** Can significantly accelerate exoplanet validation by reducing false positives and improving classification.
* **Ready-to-Use API:** Supports integration into existing NASA workflows or broader scientific tools.
* **CSV based Data Input:** Supports multi row CSV inputs for the ease of use of the users.

---

### **Creativity & Design Considerations**

* Designed a **staged fine-tuning pipeline** to adapt to different missions without losing performance.
* Applied **data-efficient training**, preserving rows for validation to maintain robust evaluation metrics.
* Engineered a **custom loss function strategy** to adjust bias and variance trade-offs per dataset.
* Built a **multi-model ensemble** as a benchmark, and then surpassed it using a fine-tuned single model.

We considered:

* Dataset domain shifts.
* Overfitting risks with small datasets.
* The need for interpretable, fast, and reliable models for scientific missions.


To run the application

1. Go to `./backend/`

2. Create a venv. 
```sh
python3 -m venv .venv
# or in windows
# python -m venv .venv
```

3. Activate venv
```sh
source venv/bin/activate
# or in windows
# .\venv\Scripts\Activate.ps1
```

4. Install dependencies
```sh
pip install -r requirements.txt
```

5. Run the backend server
```sh
uvicorn app:app --reload
```


## To run the frontend

1. Go to `./frontend/`
2. Install dependencies
```sh
npm install
```
3. Run the development server
```sh
npm run dev
```
