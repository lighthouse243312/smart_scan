# Release builds run R8. These libraries call into native code (JNI) that finds Java classes,
# fields and methods by name, so their names must survive shrinking and obfuscation — otherwise
# the native side gets null back and the image pipeline fails with errors like
# "Attempt to invoke virtual method 'java.lang.Class java.lang.Object.getClass()' on a null object".
-keep class org.opencv.** { *; }
-keep class org.tensorflow.** { *; }
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_** { *; }
# the app's own image pipeline (MethodChannel handlers, models read by reflection)
-keep class com.example.beacon_smart_scan.** { *; }
-dontwarn org.tensorflow.**
-dontwarn org.opencv.**
