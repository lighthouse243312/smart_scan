// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Chinese (`zh`).
class AppLocalizationsZh extends AppLocalizations {
  AppLocalizationsZh([String locale = 'zh']) : super(locale);

  @override
  String get appTitle => 'Smart Scan';

  @override
  String get homeHeadline => '文档整洁，\n随时分享';

  @override
  String get homeSubtitle =>
      '扫描或选择照片，应用会自动锐化、去除阴影，并可擦除手写字迹。可导出为 PDF、Word 或 PNG。';

  @override
  String get scanTitle => '扫描文档';

  @override
  String get scanSubtitle => '使用相机，自动识别边框并裁剪';

  @override
  String get libraryTitle => '从相册选择';

  @override
  String get librarySubtitle => '选择设备中已有的照片，可多选';

  @override
  String get pdfTitle => '导入 PDF';

  @override
  String get pdfSubtitle => 'PDF 文件的每一页都会作为一页处理';

  @override
  String get exportTo => '导出格式';

  @override
  String get language => '语言';

  @override
  String get systemLanguage => '跟随系统';

  @override
  String get stepCapture => '拍摄';

  @override
  String get stepProcess => '处理';

  @override
  String get stepHandwriting => '手写';

  @override
  String get stepExport => '导出';

  @override
  String pageOf(int current, int total) {
    return '第 $current/$total 页';
  }

  @override
  String get processTitle => '锐化与去阴影';

  @override
  String get rotateLeft => '向左旋转';

  @override
  String get rotateRight => '向右旋转';

  @override
  String get deletePage => '删除页面';

  @override
  String get deletePageTitle => '删除此页？';

  @override
  String deletePageBody(int page) {
    return '第 $page 页将从文档中移除。';
  }

  @override
  String get deleteThisPage => '删除此页';

  @override
  String get deleteAllPages => '删除所有页面';

  @override
  String deleteAllPagesTitle(int count) {
    return '删除全部 $count 页？';
  }

  @override
  String get deleteAllPagesBody => '所有页面都将被移除，文档也会从此设备中删除。';

  @override
  String get cancel => '取消';

  @override
  String get delete => '删除';

  @override
  String get addScanPage => '继续扫描';

  @override
  String get addLibraryPage => '从相册添加';

  @override
  String get addPdfPage => '从 PDF 文件添加';

  @override
  String get processing => '正在锐化并去除阴影…';

  @override
  String get original => '原图';

  @override
  String get processed => '已处理';

  @override
  String get exportNow => '立即导出';

  @override
  String get eraseWritingShort => '擦除字迹';

  @override
  String get handwritingTitle => '手写';

  @override
  String get hideMask => '隐藏检测区域';

  @override
  String get showMask => '显示检测区域';

  @override
  String get saveOriginal => '将原图保存到相册';

  @override
  String get methodInk => '墨水颜色';

  @override
  String get methodAi => '墨水颜色 + AI';

  @override
  String get sensitivity => '灵敏度';

  @override
  String get brushView => '查看';

  @override
  String get brushAdd => '添加擦除区域';

  @override
  String get brushKeep => '保留';

  @override
  String get skip => '跳过';

  @override
  String get eraseHandwriting => '擦除手写字迹';

  @override
  String detectingPages(int current, int total) {
    return '正在查找手写… 第 $current/$total 页';
  }

  @override
  String erasingPages(int current, int total) {
    return '正在擦除手写… 第 $current/$total 页';
  }

  @override
  String get eraseAllPages => '擦除所有页面';

  @override
  String get hintNoneInk => '未发现与印刷颜色不同的笔迹。请提高灵敏度；若笔迹与印刷颜色相同，请使用“墨水颜色 + AI”。';

  @override
  String get hintNoneAi => '未发现手写字迹。请提高灵敏度或用画笔涂抹。';

  @override
  String hintCoverage(String percent) {
    return '红色区域（占页面 $percent%）将被擦除；黄色是被字迹覆盖的印刷内容，会被重建。可用画笔添加区域或保留误标的文字；更换方式或灵敏度会重新检测。';
  }

  @override
  String get savedOriginal => '原图已保存到相册。';

  @override
  String saveFailed(String error) {
    return '保存失败：$error';
  }

  @override
  String get exportTitle => '导出文档';

  @override
  String get fileName => '文件名';

  @override
  String get formatPdf => 'PDF';

  @override
  String get formatWord => 'Word';

  @override
  String get formatPng => 'PNG 图片';

  @override
  String get formatPdfDesc => '一个文件，每页一张 A4';

  @override
  String get formatWordDesc => '可用 Word 或 Google 文档打开的 .docx 文件';

  @override
  String get formatPngDesc => '每页一张原画质图片';

  @override
  String shareFormat(String format) {
    return '分享 $format';
  }

  @override
  String get saveImages => '保存图片';

  @override
  String get newDocument => '新建文档';

  @override
  String savedImages(int count) {
    return '已将 $count 张图片保存到相册';
  }

  @override
  String actionFailed(String error) {
    return '无法完成：$error';
  }

  @override
  String get galleryPermission => '未获得相册访问权限';

  @override
  String processingFailed(String code) {
    return '图片处理失败（$code）';
  }

  @override
  String get tabScan => '扫描';

  @override
  String get tabDocuments => '文档';

  @override
  String get tabAccount => '账户';

  @override
  String get documentsTitle => '已保存的文档';

  @override
  String get documentsEmpty => '还没有文档';

  @override
  String get documentsEmptyHint => '每次扫描都会在此保存原图；您选择保存的处理版本会显示在旁边。';

  @override
  String pagesCount(int count) {
    return '$count 页';
  }

  @override
  String get editAgain => '重新编辑';

  @override
  String get share => '分享';

  @override
  String get rename => '重命名';

  @override
  String get deleteDocument => '删除文档';

  @override
  String get deleteDocumentTitle => '删除此文档？';

  @override
  String get deleteDocumentBody => '原图和处理版本将从本设备永久删除。';

  @override
  String get save => '保存';

  @override
  String get documentName => '文档名称';

  @override
  String get saveToLibrary => '保存到文档';

  @override
  String get savedToLibrary => '已将处理版本保存到文档';

  @override
  String get originalsKept => '原图会自动保存在“文档”中';

  @override
  String get accountGuest => '访客';

  @override
  String get accountGuestHint => '文档保存在本设备上。登录后可在多设备间同步（即将推出）。';

  @override
  String get signIn => '登录';

  @override
  String get comingSoon => '即将推出';

  @override
  String get settings => '设置';

  @override
  String get storage => '存储';

  @override
  String storageCount(int count) {
    return '本设备上有 $count 个文档';
  }

  @override
  String get about => '关于';

  @override
  String get version => '版本';

  @override
  String get recentDocuments => '最近';

  @override
  String get seeAll => '查看全部';

  @override
  String get noEdits => '暂无处理版本';
}
