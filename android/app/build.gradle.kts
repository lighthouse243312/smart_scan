plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.beacon_smart_scan"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.beacon_smart_scan"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        // dev harness for the image pipeline (src/androidTest), not part of the app
        testInstrumentationRunner = "com.example.beacon_smart_scan.InkHarness"
    }

    // -PharnessRelease runs InkHarness against the release (non-debuggable) build, for timings that
    // match what users get: a debuggable app runs its Kotlin pixel loops far slower
    val harnessRelease = project.hasProperty("harnessRelease")
    if (harnessRelease) testBuildType = "release"

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
            // R8 would strip the Kotlin stdlib the harness needs from the app it instruments
            if (harnessRelease) { isMinifyEnabled = false; isShrinkResources = false }
        }
    }

    androidResources {
        // .tflite is a binary flatbuffer — must ship byte-for-byte, not gzip-compressed like
        // other assets (AAPT would otherwise try to compress it, which the loader can't mmap).
        noCompress += "tflite"
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}

dependencies {
    implementation(project(":opencv"))
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    // on-device text recognition (bundled models) for restoring pen-hidden print, every script ML
    // Kit has: Latin (incl. Vietnamese), Chinese, Japanese, Korean, Devanagari
    implementation("com.google.mlkit:text-recognition:16.0.1")
    implementation("com.google.mlkit:text-recognition-chinese:16.0.1")
    implementation("com.google.mlkit:text-recognition-japanese:16.0.1")
    implementation("com.google.mlkit:text-recognition-korean:16.0.1")
    implementation("com.google.mlkit:text-recognition-devanagari:16.0.1")
}
