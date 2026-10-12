import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:beacon_smart_scan/app.dart';

void main() {
  testWidgets('App launches to the scan tab', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues({});
    await tester.pumpWidget(const SmartScanApp());
    await tester.pump();

    // the test environment's locale is English
    expect(find.text('Scan document'), findsOneWidget);
    expect(find.text('Documents'), findsOneWidget);
    expect(find.text('Account'), findsNothing);
    expect(find.text('English'), findsOneWidget); // the language chip shows the language in use
  });
}
