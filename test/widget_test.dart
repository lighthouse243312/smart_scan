import 'package:flutter_test/flutter_test.dart';

import 'package:beacon_smart_scan/app.dart';

void main() {
  testWidgets('App launches to the capture screen', (WidgetTester tester) async {
    await tester.pumpWidget(const SmartScanApp());

    expect(find.text('Quét tài liệu'), findsWidgets);
  });
}
