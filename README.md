# captcha-solver

Tested on Windows, Python 3.8
1. Install python packages from requirements.txt
2. Run ***extract_single_letter_images.py*** to extract and save single digits, labels from captcha images.(Optional, already present)
3. Run ***train_model.py*** to train the CNN model, it will generate ***captcha_model.keras*** file.(Optional, already present)
4. Run ***main.py*** for FastAPI to serve and predict captcha image api.
5. Tested on postman, ***http://127.0.0.1:8000/predict/captcha/***
6. Add/upload image file in body of API request to predict.
