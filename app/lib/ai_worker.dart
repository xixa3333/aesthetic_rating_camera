import 'dart:isolate';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

// 🌟 供外部 Isolate 呼叫的公開入口
void backgroundWorkerEntry(Map<String, dynamic> initArgs) async {
  final Uint8List modelBytes = initArgs['modelBytes'];
  final SendPort mainSendPort = initArgs['mainSendPort'];

  var options = InterpreterOptions()..threads = 4;
  Interpreter interpreter = Interpreter.fromBuffer(modelBytes, options: options);

  final ReceivePort isolateReceivePort = ReceivePort();
  mainSendPort.send(isolateReceivePort.sendPort);

  await for (var message in isolateReceivePort) {
    if (message is Map<String, dynamic>) {
      final stopwatch = Stopwatch()..start();
      try {
        img.Image parsedImage = convertYUV420(message);
        Float32List inputTensor = preprocessImage(parsedImage);
        
        var input = inputTensor.reshape([1, 512, 512, 3]);
        var output = List.filled(1, 0.0);
        
        interpreter.run(input, output);
        stopwatch.stop();

        double rawScore = output[0];
        // 1. 正規化至 0~1
        double t = ((rawScore - 1.0) / (9.0 - 1.0)).clamp(0.0, 1.0);
        // 2. 三次 Hermite 平滑過渡 S 曲線：3t^2 - 2t^3
        double smoothT = t * t * (3.0 - 2.0 * t);
        // 3. 映射回 0.0 ~ 10.0 分
        double finalScore = smoothT * 10.0;

        mainSendPort.send({
          'score': finalScore,
          'time': stopwatch.elapsedMilliseconds,
        });
      } catch (e) {
        mainSendPort.send({'score': 0.0, 'time': 0});
      }
    }
  }
}

img.Image convertYUV420(Map<String, dynamic> imageData) {
  final int width = imageData['width'];
  final int height = imageData['height'];
  final int yRowStride = imageData['y_row_stride']; 
  final int uvRowStride = imageData['uv_row_stride'];
  final int uvPixelStride = imageData['uv_pixel_stride'];
  
  final Uint8List yPlane = imageData['y_plane'];
  final Uint8List uPlane = imageData['u_plane'];
  final Uint8List vPlane = imageData['v_plane'];
  
  var img1 = img.Image(width: width, height: height);
  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final int yIndex = y * yRowStride + x; 
      final int uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
      
      if (yIndex >= yPlane.length || uvIndex >= uPlane.length || uvIndex >= vPlane.length) continue;
      
      final int yp = yPlane[yIndex];
      final int up = uPlane[uvIndex];
      final int vp = vPlane[uvIndex];
      
      int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
      int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91).round().clamp(0, 255);
      int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
      
      img1.setPixelRgba(x, y, r, g, b, 255);
    }
  }
  return img1;
}

Float32List preprocessImage(img.Image originalImage) {
  const int targetSize = 512; 
  int totalR = 0, totalG = 0, totalB = 0;
  
  final pixels = originalImage.getBytes(order: img.ChannelOrder.rgb);
  final pixelCount = originalImage.width * originalImage.height;
  
  for (int i = 0; i < pixels.length; i += 3) {
    totalR += pixels[i];
    totalG += pixels[i + 1];
    totalB += pixels[i + 2];
  }
  
  int meanR = totalR ~/ pixelCount;
  int meanG = totalG ~/ pixelCount;
  int meanB = totalB ~/ pixelCount;

  img.Image paddedImage = img.Image(width: targetSize, height: targetSize);
  img.fill(paddedImage, color: img.ColorRgba8(meanR, meanG, meanB, 255));

  double scale = targetSize / originalImage.width >= targetSize / originalImage.height
      ? targetSize / originalImage.height
      : targetSize / originalImage.width;
      
  int newW = (originalImage.width * scale).toInt();
  int newH = (originalImage.height * scale).toInt();

  img.Image resizedImage = img.copyResize(originalImage, width: newW, height: newH, interpolation: img.Interpolation.linear);

  int offsetX = (targetSize - newW) ~/ 2;
  int offsetY = (targetSize - newH) ~/ 2;
  img.compositeImage(paddedImage, resizedImage, dstX: offsetX, dstY: offsetY);

  Float32List inputTensor = Float32List(targetSize * targetSize * 3);
  int index = 0;
  final finalPixels = paddedImage.getBytes(order: img.ChannelOrder.rgb);
  
  const double imageNetMeanR = 0.485;
  const double imageNetStdR = 0.229;
  const double imageNetMeanG = 0.456;
  const double imageNetStdG = 0.224;
  const double imageNetMeanB = 0.406;
  const double imageNetStdB = 0.225;

  for (int i = 0; i < finalPixels.length; i += 3) {
    inputTensor[index++] = ((finalPixels[i] / 255.0) - imageNetMeanR) / imageNetStdR;
    inputTensor[index++] = ((finalPixels[i + 1] / 255.0) - imageNetMeanG) / imageNetStdG;
    inputTensor[index++] = ((finalPixels[i + 2] / 255.0) - imageNetMeanB) / imageNetStdB;
  }

  return inputTensor;
}