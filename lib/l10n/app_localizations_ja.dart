// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Japanese (`ja`).
class AppLocalizationsJa extends AppLocalizations {
  AppLocalizationsJa([String locale = 'ja']) : super(locale);

  @override
  String get appTitle => 'Smart Scan';

  @override
  String get homeHeadline => 'きれいな書類を、\nすぐに共有';

  @override
  String get homeSubtitle =>
      'スキャンまたは写真を選ぶと、鮮明化・影の除去を行い、手書き文字も消せます。PDF・Word・PNGで書き出せます。';

  @override
  String get scanTitle => '書類をスキャン';

  @override
  String get scanSubtitle => 'カメラで撮影し、枠を自動で検出・切り抜き';

  @override
  String get libraryTitle => 'ライブラリから選ぶ';

  @override
  String get librarySubtitle => '端末内の写真を複数まとめて選択';

  @override
  String get pdfTitle => 'PDFを読み込む';

  @override
  String get pdfSubtitle => 'PDFファイルの各ページを処理します';

  @override
  String get exportTo => '書き出し形式';

  @override
  String get language => '言語';

  @override
  String get systemLanguage => 'システムに従う';

  @override
  String get stepCapture => '撮影';

  @override
  String get stepProcess => '処理';

  @override
  String get stepHandwriting => '手書き';

  @override
  String get stepExport => '書き出し';

  @override
  String pageOf(int current, int total) {
    return '$current/$totalページ';
  }

  @override
  String get processTitle => '鮮明化と影の除去';

  @override
  String get rotateLeft => '左に回転';

  @override
  String get rotateRight => '右に回転';

  @override
  String get deletePage => 'ページを削除';

  @override
  String get deletePageTitle => 'このページを削除しますか？';

  @override
  String deletePageBody(int page) {
    return '$pageページ目が書類から削除されます。';
  }

  @override
  String get deleteThisPage => 'このページを削除';

  @override
  String get deleteAllPages => 'すべてのページを削除';

  @override
  String deleteAllPagesTitle(int count) {
    return '$countページすべてを削除しますか？';
  }

  @override
  String get deleteAllPagesBody => 'すべてのページが削除され、書類もこの端末から削除されます。';

  @override
  String get cancel => 'キャンセル';

  @override
  String get delete => '削除';

  @override
  String get addScanPage => 'ページを追加スキャン';

  @override
  String get addLibraryPage => 'ライブラリから追加';

  @override
  String get addPdfPage => 'PDFファイルから追加';

  @override
  String get processing => '鮮明化と影の除去を実行中…';

  @override
  String get original => '元の画像';

  @override
  String get processed => '処理済み';

  @override
  String get exportNow => '今すぐ書き出す';

  @override
  String get eraseWritingShort => '手書きを消す';

  @override
  String get handwritingTitle => '手書き';

  @override
  String get hideMask => '検出範囲を隠す';

  @override
  String get showMask => '検出範囲を表示';

  @override
  String get saveOriginal => '元の画像を写真に保存';

  @override
  String get methodInk => 'インクの色';

  @override
  String get methodAi => 'インクの色 + AI';

  @override
  String get sensitivity => '感度';

  @override
  String get brushView => '表示';

  @override
  String get brushAdd => '消す範囲を追加';

  @override
  String get brushKeep => '残す';

  @override
  String get skip => 'スキップ';

  @override
  String get eraseHandwriting => '手書きを消す';

  @override
  String detectingPages(int current, int total) {
    return '手書きを検出中… $current/$totalページ';
  }

  @override
  String erasingPages(int current, int total) {
    return '手書きを消去中… $current/$totalページ';
  }

  @override
  String get eraseAllPages => '全ページを消去';

  @override
  String get hintNoneInk =>
      '印刷と異なる色のペン跡が見つかりません。感度を上げるか、印刷と同じ色のペンには「インクの色 + AI」を使ってください。';

  @override
  String get hintNoneAi => '手書きが見つかりません。感度を上げるか、ブラシで塗ってください。';

  @override
  String hintCoverage(String percent) {
    return '赤い範囲（ページの$percent%）が消去されます。黄色は手書きが重なった印刷部分で、復元されます。ブラシで範囲を追加したり、誤って塗られた文字を残したりできます。方法や感度を変えると最初から検出し直します。';
  }

  @override
  String get savedOriginal => '元の画像を写真に保存しました。';

  @override
  String saveFailed(String error) {
    return '保存に失敗しました：$error';
  }

  @override
  String get exportTitle => '書類を書き出す';

  @override
  String get fileName => 'ファイル名';

  @override
  String get formatPdf => 'PDF';

  @override
  String get formatWord => 'Word';

  @override
  String get formatPng => 'PNG画像';

  @override
  String get formatPdfDesc => '1つのファイル、1ページごとにA4';

  @override
  String get formatWordDesc => 'WordやGoogleドキュメントで開ける.docxファイル';

  @override
  String get formatPngDesc => 'ページごとに高画質の画像1枚';

  @override
  String shareFormat(String format) {
    return '$formatを共有';
  }

  @override
  String get saveImages => '画像を保存';

  @override
  String get newDocument => '新しい書類';

  @override
  String savedImages(int count) {
    return '$count枚の画像をギャラリーに保存しました';
  }

  @override
  String actionFailed(String error) {
    return '実行できませんでした：$error';
  }

  @override
  String get galleryPermission => '写真ライブラリへのアクセスが許可されていません';

  @override
  String processingFailed(String code) {
    return '画像処理に失敗しました（$code）';
  }

  @override
  String get tabScan => 'スキャン';

  @override
  String get tabDocuments => '書類';

  @override
  String get tabAccount => 'アカウント';

  @override
  String get documentsTitle => '保存した書類';

  @override
  String get documentsEmpty => '書類はまだありません';

  @override
  String get documentsEmptyHint => 'スキャンごとに元の画像がここに保存され、保存した処理済み版も並んで表示されます。';

  @override
  String pagesCount(int count) {
    return '$countページ';
  }

  @override
  String get editAgain => 'もう一度編集';

  @override
  String get share => '共有';

  @override
  String get rename => '名前を変更';

  @override
  String get deleteDocument => '書類を削除';

  @override
  String get deleteDocumentTitle => 'この書類を削除しますか？';

  @override
  String get deleteDocumentBody => '元の画像と処理済み版がこの端末から完全に削除されます。';

  @override
  String get save => '保存';

  @override
  String get documentName => '書類名';

  @override
  String get saveToLibrary => '書類に保存';

  @override
  String get savedToLibrary => '処理済み版を書類に保存しました';

  @override
  String get originalsKept => '元の画像は「書類」に自動保存されます';

  @override
  String get accountGuest => 'ゲスト';

  @override
  String get accountGuestHint => '書類はこの端末に保存されています。ログインすると端末間で同期できます（近日公開）。';

  @override
  String get signIn => 'ログイン';

  @override
  String get comingSoon => '近日公開';

  @override
  String get settings => '設定';

  @override
  String get storage => 'ストレージ';

  @override
  String storageCount(int count) {
    return 'この端末に$count件の書類';
  }

  @override
  String get about => 'アプリについて';

  @override
  String get version => 'バージョン';

  @override
  String get recentDocuments => '最近';

  @override
  String get seeAll => 'すべて表示';

  @override
  String get noEdits => '処理済み版はまだありません';
}
