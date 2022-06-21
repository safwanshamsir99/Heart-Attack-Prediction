# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:40:38 2022

@author: safwanshamsir99
"""

import streamlit as st
import os
import numpy as np
import pickle

MODEL_PATH = os.path.join(os.getcwd(),'best_model_heart.pkl')

with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)

# X = df.loc[:,['age','cp','thalachh','oldpeak','thall']]

#%% STREAMLIT
with st.form("Patient's Form"):
    st.title("Heart Attack Prediction")
    st.video("https://youtu.be/jvOU4Do4xZ8", format="video/mp4") 
    # credit video: "How Old Is Your Heart? Learn Your Heart Age!" By CDC YouTube channel
    st.header("What is your heart age? Let's check it out!")
    age = int(st.number_input("Key in your age: "))
    st.write("For chest pain type: 0 = Typical angina,",
             "\n1 = Atypical angina,",
             "\n2 = Non-anginal pain,",
             "\n3 = Asymptomatic")
    cp = int(st.number_input("Key in your chest pain type: "))
    thalachh = int(st.number_input("Key in your maximum heart rate achieved: "))
    oldpeak = int(st.number_input("Key in your ST depression value: "))
    st.write("For thallasemia: 1 = Fixed defect,",
             "\n2 = Normal,",
             "\n3 = Reverseable defect")
    thall = int(st.radio("Do you have thallasemia?", (1,2,3)))

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Age:",age,
                 "Chest pain type:",cp,
                 "Maximum heart rate achieved:",thalachh,
                 "ST depression value:",oldpeak,
                 "Thallasemia:",thall)
        temp = np.expand_dims([age,cp,thalachh,oldpeak,thall], axis=0)
        outcome = model.predict(temp)
        
        outcome_dict = {0:'Less chance of heart attack',
                        1:'More chance of heart attack'}
        
        if outcome == 1:
            st.snow()
            st.markdown('**High possibility** to get a heart attack!')
            st.write("Please change your lifestyle, make your heart healthy and young!")
            st.image("https://cdn.shopify.com/s/files/1/1060/9112/products/109r.jpeg?v=1579804727")
            # Credit pic: foodandhealth.com website
        else:
            st.balloons()
            st.write("Voila, you have a young heart age. Please keep your healthy lifestyle!")
            st.image("https://www.caring.com/graphics/caring-heart-healthy-tips.jpg")
            # Credit pic: caring.com/graphics website
        
