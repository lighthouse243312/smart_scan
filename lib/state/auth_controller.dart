import 'package:flutter/foundation.dart';

/// The signed-in account, if any. Sign-in does not exist yet: everyone is the local guest. When
/// it lands, [signIn] / [signOut] switch [userId] and every per-user store follows (see
/// DocumentLibrary).
class AppUser {
  const AppUser({required this.id, this.displayName, this.email});

  final String id;
  final String? displayName;
  final String? email;
}

class AuthController extends ChangeNotifier {
  static const guestId = 'guest';

  AppUser? _user;
  AppUser? get user => _user;
  bool get isSignedIn => _user != null;

  /// Owner id for per-user storage.
  String get userId => _user?.id ?? guestId;

  /// Whether a sign-in provider is wired up (false until one is added).
  bool get signInAvailable => false;

  Future<void> signIn(AppUser user) async {
    _user = user;
    notifyListeners();
  }

  Future<void> signOut() async {
    _user = null;
    notifyListeners();
  }
}
