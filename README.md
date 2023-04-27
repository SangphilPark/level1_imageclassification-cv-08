# level1_imageclassification-cv-08 (오기의 아이들)

---

구성원 : 김보경, 박상필, 이태순, 임현명, 천지은

### Dependencies

---

이 모델은 Ubuntu 18.04.5 LTS, Tesla v100 32GB의 환경에서 작성 및 테스트 되었습니다. 또 모델 실행에는 다음과 같은 외부 라이브러리가 필요합니다.

- torch == 1.7.1
- torchvision==0.8.2
- pandas~=1.2.0
- scikit-learn~=0.24.1
- matplotlib==3.5.1
- numpy~=1.21.5
- python-dotenv~=0.16.0
- Pillow~=7.2.0
- sklearn~=0.0
- timm==0.6.13

Install dependencies: `pip3 install -r requirements.txt`

### Dataset

---

이 코드는 부스트캠프에서 제공하는 데이터셋으로 다음과 같은 구성을 따릅니다. 

- 전체 사람 명 수 : 4500
- 한 사람당 사진의 개수 : 7 (마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장)
    
    ```python
    data
    |-- eval
    |   |-- images
    |   |   `-- eval_dataset
    |   `-- info.csv
    `-- train
        |-- imgaes_gen
        |   `-- ID_gender_race_age
        |       |-- incorrect_mask
        |       |-- mask1
        |       |-- mask2
        |       |-- mask3
        |       |-- mask4
        |       |-- mask5
        |       |-- normal
        `-- train.csv
    ```
    
- 이미지 크기 : (384, 512)
- 분류 클래스 : 마스크 착용 여부, 성별, 나이를 기준으로 총 18개의 클래스 존재
- 프로젝트 구조
    
    ```python
    project/
    |-- data_gen_rembg.ipynb
    |-- inference.py
    |-- inference_multiclass.py
    |-- loss.py
    |-- model.py
    |-- rembg_dataset.py
    |-- requirements.txt
    |-- sample_submission.ipynb
    |-- skf_train.py
    |-- skf_train_multiclass.py
    |-- train.py
    |-- train_multiclass.py
    `-- train_optuna.py
    ```
    

### Train

---

모든 모델을 처음부터 학습하기 위해선 다음 명령어들을 사용합니다.

**기본 모델**

```python
python train.py \
	--

```

**Multiclass로 학습**

```python
python train_multiclass.py \ 

```

**Stratified k-fold로 학습**

```python
python skf_train.py \ 
```

**optuna를 활용하여 학습**