// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Vietnamese (`vi`).
class AppLocalizationsVi extends AppLocalizations {
  AppLocalizationsVi([String locale = 'vi']) : super(locale);

  @override
  String get appTitle => 'Smart Scan';

  @override
  String get homeHeadline => 'Tài liệu gọn gàng,\nsẵn sàng chia sẻ';

  @override
  String get homeSubtitle =>
      'Quét hoặc chọn ảnh, app tự làm nét, xoá bóng và có thể xoá chữ viết tay. Xuất ra PDF, Word hoặc ảnh PNG.';

  @override
  String get scanTitle => 'Quét tài liệu';

  @override
  String get scanSubtitle => 'Dùng camera, tự nhận khung và cắt trang';

  @override
  String get libraryTitle => 'Chọn từ thư viện';

  @override
  String get librarySubtitle => 'Lấy ảnh có sẵn trong máy, chọn được nhiều ảnh';

  @override
  String get pdfTitle => 'Nhập file PDF';

  @override
  String get pdfSubtitle => 'Mỗi trang của file PDF thành một trang để xử lý';

  @override
  String get exportTo => 'Xuất ra';

  @override
  String get language => 'Ngôn ngữ';

  @override
  String get systemLanguage => 'Theo hệ thống';

  @override
  String get stepCapture => 'Chụp';

  @override
  String get stepProcess => 'Xử lý';

  @override
  String get stepHandwriting => 'Chữ viết tay';

  @override
  String get stepExport => 'Xuất';

  @override
  String pageOf(int current, int total) {
    return 'Trang $current/$total';
  }

  @override
  String get processTitle => 'Làm nét & xoá bóng';

  @override
  String get rotateLeft => 'Xoay trái';

  @override
  String get rotateRight => 'Xoay phải';

  @override
  String get deletePage => 'Xoá trang';

  @override
  String get deletePageTitle => 'Xoá trang này?';

  @override
  String deletePageBody(int page) {
    return 'Trang $page sẽ bị bỏ khỏi tài liệu.';
  }

  @override
  String get deleteThisPage => 'Xoá trang này';

  @override
  String get deleteAllPages => 'Xoá tất cả trang';

  @override
  String deleteAllPagesTitle(int count) {
    return 'Xoá tất cả $count trang?';
  }

  @override
  String get deleteAllPagesBody =>
      'Mọi trang sẽ bị xoá và tài liệu bị xoá khỏi thiết bị này.';

  @override
  String get cancel => 'Huỷ';

  @override
  String get delete => 'Xoá';

  @override
  String get addScanPage => 'Quét thêm trang';

  @override
  String get addLibraryPage => 'Chọn thêm từ thư viện';

  @override
  String get addPdfPage => 'Thêm từ file PDF';

  @override
  String get processing => 'Đang làm nét và xoá bóng…';

  @override
  String get original => 'Ảnh gốc';

  @override
  String get processed => 'Đã xử lý';

  @override
  String get exportNow => 'Xuất ngay';

  @override
  String get eraseWritingShort => 'Xoá chữ viết';

  @override
  String get handwritingTitle => 'Chữ viết tay';

  @override
  String get hideMask => 'Ẩn vùng phát hiện';

  @override
  String get showMask => 'Hiện vùng phát hiện';

  @override
  String get saveOriginal => 'Lưu ảnh gốc vào Ảnh';

  @override
  String get methodInk => 'Màu mực';

  @override
  String get methodAi => 'Màu mực + AI';

  @override
  String get sensitivity => 'Độ nhạy';

  @override
  String get brushView => 'Xem';

  @override
  String get brushAdd => 'Thêm vùng xoá';

  @override
  String get brushKeep => 'Giữ lại';

  @override
  String get skip => 'Bỏ qua';

  @override
  String get eraseHandwriting => 'Xoá chữ viết tay';

  @override
  String detectingPages(int current, int total) {
    return 'Đang tìm chữ viết tay… trang $current/$total';
  }

  @override
  String erasingPages(int current, int total) {
    return 'Đang xoá chữ viết tay… trang $current/$total';
  }

  @override
  String get eraseAllPages => 'Xoá tất cả trang';

  @override
  String get hintNoneInk =>
      'Không thấy nét bút khác màu mực in. Thử tăng độ nhạy; bút cùng màu mực in thì dùng \"Màu mực + AI\".';

  @override
  String get hintNoneAi =>
      'Không thấy chữ viết tay. Thử tăng độ nhạy hoặc tô thêm bằng cọ.';

  @override
  String hintCoverage(String percent) {
    return 'Vùng đỏ ($percent% trang) sẽ bị xoá; chỗ vàng là chữ in bị viết đè, sẽ được dựng lại. Dùng cọ để thêm vùng hoặc giữ lại chữ bị tô nhầm; đổi cách hoặc độ nhạy sẽ phát hiện lại từ đầu.';
  }

  @override
  String get savedOriginal => 'Đã lưu ảnh gốc vào Ảnh.';

  @override
  String saveFailed(String error) {
    return 'Lưu thất bại: $error';
  }

  @override
  String get exportTitle => 'Xuất tài liệu';

  @override
  String get fileName => 'Tên tệp';

  @override
  String get formatPdf => 'PDF';

  @override
  String get formatWord => 'Word';

  @override
  String get formatPng => 'Ảnh PNG';

  @override
  String get formatPdfDesc => 'Một tệp, mỗi trang một khổ A4';

  @override
  String get formatWordDesc => 'Tệp .docx mở được bằng Word, Google Docs';

  @override
  String get formatPngDesc => 'Mỗi trang một ảnh chất lượng gốc';

  @override
  String shareFormat(String format) {
    return 'Chia sẻ $format';
  }

  @override
  String get saveImages => 'Lưu ảnh';

  @override
  String get newDocument => 'Tài liệu mới';

  @override
  String savedImages(int count) {
    return 'Đã lưu $count ảnh vào thư viện';
  }

  @override
  String actionFailed(String error) {
    return 'Không thực hiện được: $error';
  }

  @override
  String get galleryPermission => 'Chưa cấp quyền truy cập thư viện ảnh';

  @override
  String processingFailed(String code) {
    return 'Xử lý ảnh thất bại ($code)';
  }

  @override
  String get tabScan => 'Quét';

  @override
  String get tabDocuments => 'Tài liệu';

  @override
  String get tabAccount => 'Tài khoản';

  @override
  String get documentsTitle => 'Tài liệu đã lưu';

  @override
  String get documentsEmpty => 'Chưa có tài liệu nào';

  @override
  String get documentsEmptyHint =>
      'Mỗi lần quét đều được lưu bản gốc tại đây; bản đã xử lý bạn chọn lưu sẽ nằm cạnh bản gốc.';

  @override
  String pagesCount(int count) {
    return '$count trang';
  }

  @override
  String get editAgain => 'Chỉnh sửa lại';

  @override
  String get share => 'Chia sẻ';

  @override
  String get rename => 'Đổi tên';

  @override
  String get deleteDocument => 'Xoá tài liệu';

  @override
  String get deleteDocumentTitle => 'Xoá tài liệu này?';

  @override
  String get deleteDocumentBody =>
      'Bản gốc và bản đã xử lý sẽ bị xoá vĩnh viễn khỏi máy.';

  @override
  String get save => 'Lưu';

  @override
  String get documentName => 'Tên tài liệu';

  @override
  String get saveToLibrary => 'Lưu vào tài liệu';

  @override
  String get savedToLibrary => 'Đã lưu bản đã xử lý vào tài liệu';

  @override
  String get originalsKept => 'Bản gốc được lưu tự động trong Tài liệu';

  @override
  String get accountGuest => 'Khách';

  @override
  String get accountGuestHint =>
      'Tài liệu đang lưu trên máy này. Đăng nhập để đồng bộ giữa các thiết bị (sắp có).';

  @override
  String get signIn => 'Đăng nhập';

  @override
  String get comingSoon => 'Sắp có';

  @override
  String get settings => 'Cài đặt';

  @override
  String get storage => 'Lưu trữ';

  @override
  String storageCount(int count) {
    return '$count tài liệu trên máy này';
  }

  @override
  String get about => 'Giới thiệu';

  @override
  String get version => 'Phiên bản';

  @override
  String get recentDocuments => 'Gần đây';

  @override
  String get seeAll => 'Xem tất cả';

  @override
  String get noEdits => 'Chưa có bản đã xử lý';
}
