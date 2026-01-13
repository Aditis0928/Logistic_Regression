{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b3f590aa-188f-47ab-afcf-ebb702a4f9e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Load trained model\n",
    "with open(\"model.pkl\", \"rb\") as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "st.title(\"Logistic Regression Deployment\")\n",
    "\n",
    "st.write(\"Enter values to predict the output\")\n",
    "\n",
    "feature1 = st.number_input(\"Feature 1\", value=0.0)\n",
    "feature2 = st.number_input(\"Feature 2\", value=0.0)\n",
    "\n",
    "input_data = np.array([[feature1, feature2]])\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    prediction = model.predict(input_data)\n",
    "    st.success(f\"Prediction result\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
