# meta-gan

* GAN의 가중치를 (또 다른) GAN이 만들어 내게 할 수는 없을까? [참고](https://www.facebook.com/groups/TensorFlowKR/permalink/623980551276340/)
* Tensorflow 사용
* 실험 내용
  1. 간단한 GAN을 만들고 MNIST 데이터셋의 10 숫자 중 하나의 숫자에 대해서만 학습 진행
  2. 숫자를 바꿔가며 1. 을 반복하고 매번 Generator의 가중치를 별도로 저장
  3. 모아진 Generator의 가중치를 하나의 데이터셋으로 보고, 가중치를 생성하는 Meta GAN 학습
  4. Meta GAN 학습이 끝나면 Meta GAN이 생성해 낸 가중치를 기존 간단한 GAN 모델에 적용(load)
  5. 생성된 가중치가 적용된 GAN이 그럴듯한 숫자 이미지를 생성해 내는지 확인
* 학습에 걸리는 시간이 간단한 GAN에 비해 매우 길어졌지만(약 5000에폭 소요), 숫자처럼 보이는 이미지를 만들어 낸다는 결과를 확인
