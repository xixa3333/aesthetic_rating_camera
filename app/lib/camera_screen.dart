import 'dart:async';
import 'dart:isolate';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:gal/gal.dart';
import 'main.dart';      // 拿 globalCameras
import 'ai_worker.dart'; // 拿 backgroundWorkerEntry

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  bool _isCameraInitialized = false;
  bool _isIsolateReady = false;
  
  bool _isProcessing = false;
  
  double _baseZoomLevel = 1.0;
  double _minAvailableZoom = 1.0;
  double _maxAvailableZoom = 1.0;
  double _currentExposureOffset = 0.0;
  double _minAvailableExposure = 0.0;
  double _maxAvailableExposure = 0.0;

  final ValueNotifier<double> _scoreNotifier = ValueNotifier<double>(1.0);
  final ValueNotifier<int> _timeNotifier = ValueNotifier<int>(0);
  final ValueNotifier<double> _zoomNotifier = ValueNotifier<double>(1.0);

  Isolate? _backgroundIsolate;
  SendPort? _backgroundSendPort;
  final ReceivePort _mainReceivePort = ReceivePort();

  @override
  void initState() {
    super.initState();
    _startBackgroundIsolate();
    _initCamera();
  }

  void _startBackgroundIsolate() async {
    try {
      ByteData modelData = await rootBundle.load('assets/mobilenet_512_simp_float16.tflite');
      Uint8List modelBytes = modelData.buffer.asUint8List();

      // 呼叫 ai_worker.dart 的公開函數
      _backgroundIsolate = await Isolate.spawn(backgroundWorkerEntry, {
        'modelBytes': modelBytes,
        'mainSendPort': _mainReceivePort.sendPort,
      });

      _mainReceivePort.listen((message) {
        if (message is SendPort) {
          _backgroundSendPort = message;
          setState(() => _isIsolateReady = true);
        } else if (message is Map<String, dynamic>) {
          _scoreNotifier.value = message['score'];
          _timeNotifier.value = message['time'];
          _isProcessing = false; 
        }
      });
    } catch (e) {
      print("❌ 啟動背景大腦失敗: $e");
    }
  }

  Uint8List _fastMemoryCopy(Uint8List source) {
    final copy = Uint8List(source.length);
    copy.setRange(0, source.length, source);
    return copy;
  }

  void _initCamera() {
    _controller = CameraController(
      globalCameras[0], 
      ResolutionPreset.medium, 
      enableAudio: false,
    );
    
    _controller.initialize().then((_) async {
      if (!mounted) return;
      
      _maxAvailableZoom = await _controller.getMaxZoomLevel();
      _minAvailableZoom = await _controller.getMinZoomLevel();
      _maxAvailableExposure = await _controller.getMaxExposureOffset();
      _minAvailableExposure = await _controller.getMinExposureOffset();

      setState(() => _isCameraInitialized = true);

      _controller.startImageStream((CameraImage image) {
        if (!_isIsolateReady || _isProcessing) return;

        _isProcessing = true; 

        try {
          final Map<String, dynamic> safeImageData = {
            'width': image.width,
            'height': image.height,
            'y_plane': _fastMemoryCopy(image.planes[0].bytes),
            'u_plane': _fastMemoryCopy(image.planes[1].bytes),
            'v_plane': _fastMemoryCopy(image.planes[2].bytes),
            'y_row_stride': image.planes[0].bytesPerRow,
            'uv_row_stride': image.planes[1].bytesPerRow,
            'uv_pixel_stride': image.planes[1].bytesPerPixel,
          };

          _backgroundSendPort?.send(safeImageData);
        } catch (e) {
          _isProcessing = false; 
        }
      });
    });
  }

  Future<void> _takePicture() async {
    if (!_controller.value.isInitialized) return;
    try {
      bool hasAccess = await Gal.hasAccess();
      if (!hasAccess) {
        hasAccess = await Gal.requestAccess();
        if (!hasAccess) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('❌ 存取被拒：請開啟相簿權限！'), backgroundColor: Colors.red),
            );
          }
          return;
        }
      }

      final XFile file = await _controller.takePicture();
      await Gal.putImage(file.path, album: 'SmartCamera');
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('📸 照片已存入 [SmartCamera]！分數：${_scoreNotifier.value.toStringAsFixed(1)}'),
            backgroundColor: Colors.green,
            duration: const Duration(seconds: 2),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('❌ 儲存失敗: $e'), backgroundColor: Colors.red),
        );
      }
    }
  }

  void _onViewFinderTap(TapDownDetails details, BoxConstraints constraints) {
    final Offset offset = Offset(
      details.localPosition.dx / constraints.maxWidth,
      details.localPosition.dy / constraints.maxHeight,
    );
    _controller.setFocusPoint(offset);
    _controller.setExposurePoint(offset);
  }

  @override
  void dispose() {
    _controller.dispose();
    _backgroundIsolate?.kill(priority: Isolate.immediate);
    _mainReceivePort.close();
    _scoreNotifier.dispose();
    _timeNotifier.dispose();
    _zoomNotifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized || !_isIsolateReady) {
      return const Scaffold(body: Center(child: CircularProgressIndicator(color: Colors.greenAccent)));
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            return Stack(
              fit: StackFit.expand,
              children: [
                GestureDetector(
                  onTapDown: (details) => _onViewFinderTap(details, constraints),
                  onScaleStart: (details) => _baseZoomLevel = _zoomNotifier.value,
                  onScaleUpdate: (details) async {
                    double zoomed = (_baseZoomLevel * details.scale).clamp(_minAvailableZoom, _maxAvailableZoom);
                    if (zoomed != _zoomNotifier.value) {
                      _zoomNotifier.value = zoomed; 
                      await _controller.setZoomLevel(zoomed); 
                    }
                  },
                  child: Center(child: CameraPreview(_controller)),
                ),

                Positioned(
                  top: 20, right: 20,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.65), 
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.white24, width: 1),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        const Text('構圖評分', style: TextStyle(fontSize: 10, color: Colors.grey, fontWeight: FontWeight.bold)),
                        const SizedBox(height: 2),
                        ValueListenableBuilder<double>(
                          valueListenable: _scoreNotifier,
                          builder: (context, score, child) {
                            return Text(
                              score.toStringAsFixed(2),
                              style: const TextStyle(fontSize: 32, fontWeight: FontWeight.bold, color: Colors.greenAccent, fontFamily: 'monospace'),
                            );
                          },
                        ),
                        ValueListenableBuilder<int>(
                          valueListenable: _timeNotifier,
                          builder: (context, time, child) => Text('$time ms', style: TextStyle(fontSize: 9, color: Colors.white.withOpacity(0.4))),
                        ),
                      ],
                    ),
                  ),
                ),

                Positioned(
                  left: 10, top: constraints.maxHeight * 0.25,
                  child: Container(
                    height: constraints.maxHeight * 0.4,
                    decoration: BoxDecoration(color: Colors.black38, borderRadius: BorderRadius.circular(20)),
                    child: RotatedBox(
                      quarterTurns: 3, 
                      child: Slider(
                        value: _currentExposureOffset,
                        min: _minAvailableExposure,
                        max: _maxAvailableExposure,
                        activeColor: Colors.amberAccent,
                        inactiveColor: Colors.white30,
                        onChanged: (value) async {
                          setState(() => _currentExposureOffset = value);
                          await _controller.setExposureOffset(value); 
                        },
                      ),
                    ),
                  ),
                ),

                Positioned(
                  bottom: 30, left: 0, right: 0,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ValueListenableBuilder<double>(
                        valueListenable: _zoomNotifier,
                        builder: (context, zoom, child) => Text('${zoom.toStringAsFixed(1)}x', style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white)),
                      ),
                      GestureDetector(
                        onTap: _takePicture,
                        child: Container(
                          width: 76, height: 76,
                          decoration: BoxDecoration(shape: BoxShape.circle, border: Border.all(color: Colors.white, width: 4)),
                          child: Container(margin: const EdgeInsets.all(4), decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle)),
                        ),
                      ),
                      const Icon(Icons.wb_sunny_outlined, color: Colors.amberAccent),
                    ],
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}