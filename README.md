# preprocessing-CV
Keras Resnet preprocessing and MobileONE preprocessing

원본 `util::preprocess_image` 함수는 cropped 이미지를 BGR→RGB로 변환한 후 픽셀값을 \[0,1] 범위로 스케일링하고, ImageNet용 mean/std로 표준화한다.

> “cv::cvtColor(cropped, rgb\_image, cv::COLOR\_BGR2RGB);”
> “rgb\_image.convertTo(float\_image, CV\_32FC3, 1.0 / 255.0);”
> “const float mean\[3] = {0.485f, 0.456f, 0.406f}; const float std\[3] = {0.229f, 0.224f, 0.225f};”
> (util.hpp, util::preprocess\_image 내)

이 전처리 방식은 TensorFlow/Keras에서 `preprocess_input(mode='tf')`를 사용하거나 PyTorch torchvision ResNet 계열 모델이 기대하는 형식이다. (추론; mean/std 방식이 해당 프레임워크 표준임을 토대로 함)

수정된 `preprocess_image_resnet` 함수는 BGR 순서를 유지한 채 raw float 픽셀값에서 Caffe ImageNet 평균값 `{103.939f, 116.779f, 123.68f}`을 채널별로 뺀다.

> “const float mean\[3] = {103.939f, 116.779f, 123.68f};”
> (위 코드는 사용자 제공 코드)

이 전처리 방식은 Caffe 기반 ResNet50 또는 Keras에서 `preprocess_input(mode='caffe')`를 사용하는 모델이 기대하는 입력 형식이다. (추론; Caffe ImageNet 평균값 사용 기준)
