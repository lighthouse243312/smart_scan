Pod::Spec.new do |s|
  s.name             = 'opencv2'
  s.version          = '4.10.0'
  s.summary          = 'Vendored official OpenCV iOS framework (opencv.org release, not the stale CocoaPods trunk OpenCV2 pod). Pinned below 5.0.0: that release'\
                        's DNN module needs an ONNX Runtime/MLAS symbol missing for device arm64 (see ImageProcessingOpenCV.mm).'
  s.homepage         = 'https://opencv.org'
  s.license          = { :type => 'Apache-2.0' }
  s.author           = 'OpenCV team'
  s.source           = { :path => '.' }
  s.platform         = :ios, '15.5'
  s.vendored_frameworks = '../Frameworks/opencv2.framework'
  s.requires_arc     = true
  s.libraries        = 'c++'
end
