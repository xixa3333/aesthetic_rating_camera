import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'camera_screen.dart'; // 引入外場畫面

// 全域相機實體（供 camera_screen.dart 存取）
late List<CameraDescription> globalCameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  globalCameras = await availableCameras();
  runApp(const SmartCameraApp());
}

class SmartCameraApp extends StatelessWidget {
  const SmartCameraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const CameraScreen(),
    );
  }
}