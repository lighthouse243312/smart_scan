// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Korean (`ko`).
class AppLocalizationsKo extends AppLocalizations {
  AppLocalizationsKo([String locale = 'ko']) : super(locale);

  @override
  String get appTitle => 'Smart Scan';

  @override
  String get homeHeadline => '깔끔한 문서,\n바로 공유하세요';

  @override
  String get homeSubtitle =>
      '스캔하거나 사진을 고르면 선명하게 하고 그림자를 없애며 손글씨도 지울 수 있습니다. PDF, Word, PNG로 내보낼 수 있습니다.';

  @override
  String get scanTitle => '문서 스캔';

  @override
  String get scanSubtitle => '카메라로 촬영하면 테두리를 자동으로 찾아 자릅니다';

  @override
  String get libraryTitle => '라이브러리에서 선택';

  @override
  String get librarySubtitle => '기기에 있는 사진을 여러 장 선택';

  @override
  String get pdfTitle => 'PDF 가져오기';

  @override
  String get pdfSubtitle => 'PDF 파일의 각 페이지를 처리합니다';

  @override
  String get exportTo => '내보내기 형식';

  @override
  String get language => '언어';

  @override
  String get systemLanguage => '시스템 설정';

  @override
  String get stepCapture => '촬영';

  @override
  String get stepProcess => '처리';

  @override
  String get stepHandwriting => '손글씨';

  @override
  String get stepExport => '내보내기';

  @override
  String pageOf(int current, int total) {
    return '$current/$total 페이지';
  }

  @override
  String get processTitle => '선명하게 · 그림자 제거';

  @override
  String get rotateLeft => '왼쪽으로 회전';

  @override
  String get rotateRight => '오른쪽으로 회전';

  @override
  String get deletePage => '페이지 삭제';

  @override
  String get deletePageTitle => '이 페이지를 삭제할까요?';

  @override
  String deletePageBody(int page) {
    return '$page페이지가 문서에서 제거됩니다.';
  }

  @override
  String get deleteThisPage => '이 페이지 삭제';

  @override
  String get deleteAllPages => '모든 페이지 삭제';

  @override
  String deleteAllPagesTitle(int count) {
    return '$count페이지를 모두 삭제할까요?';
  }

  @override
  String get deleteAllPagesBody => '모든 페이지가 제거되고 문서가 이 기기에서 삭제됩니다.';

  @override
  String get cancel => '취소';

  @override
  String get delete => '삭제';

  @override
  String get addScanPage => '페이지 더 스캔';

  @override
  String get addLibraryPage => '라이브러리에서 추가';

  @override
  String get addPdfPage => 'PDF 파일에서 추가';

  @override
  String get processing => '선명하게 하고 그림자를 제거하는 중…';

  @override
  String get original => '원본';

  @override
  String get processed => '처리됨';

  @override
  String get exportNow => '바로 내보내기';

  @override
  String get eraseWritingShort => '손글씨 지우기';

  @override
  String get handwritingTitle => '손글씨';

  @override
  String get hideMask => '감지 영역 숨기기';

  @override
  String get showMask => '감지 영역 표시';

  @override
  String get saveOriginal => '원본을 사진에 저장';

  @override
  String get methodInk => '잉크 색';

  @override
  String get methodAi => '잉크 색 + AI';

  @override
  String get sensitivity => '민감도';

  @override
  String get brushView => '보기';

  @override
  String get brushAdd => '지울 영역 추가';

  @override
  String get brushKeep => '유지';

  @override
  String get skip => '건너뛰기';

  @override
  String get eraseHandwriting => '손글씨 지우기';

  @override
  String detectingPages(int current, int total) {
    return '손글씨 찾는 중… $current/$total페이지';
  }

  @override
  String erasingPages(int current, int total) {
    return '손글씨 지우는 중… $current/$total페이지';
  }

  @override
  String get eraseAllPages => '모든 페이지 지우기';

  @override
  String get hintNoneInk =>
      '인쇄와 다른 색의 펜 자국을 찾지 못했습니다. 민감도를 높이거나, 인쇄와 같은 색의 펜이면 \"잉크 색 + AI\"를 사용하세요.';

  @override
  String get hintNoneAi => '손글씨를 찾지 못했습니다. 민감도를 높이거나 브러시로 칠해 주세요.';

  @override
  String hintCoverage(String percent) {
    return '빨간 영역(페이지의 $percent%)이 지워집니다. 노란색은 손글씨에 덮인 인쇄 부분으로, 복원됩니다. 브러시로 영역을 추가하거나 잘못 표시된 글자를 유지할 수 있으며, 방법이나 민감도를 바꾸면 처음부터 다시 감지합니다.';
  }

  @override
  String get savedOriginal => '원본을 사진에 저장했습니다.';

  @override
  String saveFailed(String error) {
    return '저장 실패: $error';
  }

  @override
  String get exportTitle => '문서 내보내기';

  @override
  String get fileName => '파일 이름';

  @override
  String get formatPdf => 'PDF';

  @override
  String get formatWord => 'Word';

  @override
  String get formatPng => 'PNG 이미지';

  @override
  String get formatPdfDesc => '파일 하나, 페이지마다 A4 한 장';

  @override
  String get formatWordDesc => 'Word나 Google 문서에서 여는 .docx 파일';

  @override
  String get formatPngDesc => '페이지마다 원본 화질 이미지 한 장';

  @override
  String shareFormat(String format) {
    return '$format 공유';
  }

  @override
  String get saveImages => '이미지 저장';

  @override
  String get newDocument => '새 문서';

  @override
  String savedImages(int count) {
    return '이미지 $count장을 갤러리에 저장했습니다';
  }

  @override
  String actionFailed(String error) {
    return '실행할 수 없습니다: $error';
  }

  @override
  String get galleryPermission => '사진 라이브러리 접근 권한이 없습니다';

  @override
  String processingFailed(String code) {
    return '이미지 처리 실패 ($code)';
  }

  @override
  String get tabScan => '스캔';

  @override
  String get tabDocuments => '문서';

  @override
  String get tabAccount => '계정';

  @override
  String get documentsTitle => '저장된 문서';

  @override
  String get documentsEmpty => '아직 문서가 없습니다';

  @override
  String get documentsEmptyHint => '스캔할 때마다 원본이 여기에 저장되고, 저장한 처리본이 옆에 표시됩니다.';

  @override
  String pagesCount(int count) {
    return '$count페이지';
  }

  @override
  String get editAgain => '다시 편집';

  @override
  String get share => '공유';

  @override
  String get rename => '이름 변경';

  @override
  String get deleteDocument => '문서 삭제';

  @override
  String get deleteDocumentTitle => '이 문서를 삭제할까요?';

  @override
  String get deleteDocumentBody => '원본과 처리본이 이 기기에서 영구적으로 삭제됩니다.';

  @override
  String get save => '저장';

  @override
  String get documentName => '문서 이름';

  @override
  String get saveToLibrary => '문서에 저장';

  @override
  String get savedToLibrary => '처리본을 문서에 저장했습니다';

  @override
  String get originalsKept => '원본은 \'문서\'에 자동으로 저장됩니다';

  @override
  String get accountGuest => '게스트';

  @override
  String get accountGuestHint =>
      '문서는 이 기기에 저장됩니다. 로그인하면 기기 간 동기화가 가능합니다(곧 제공).';

  @override
  String get signIn => '로그인';

  @override
  String get comingSoon => '곧 제공';

  @override
  String get settings => '설정';

  @override
  String get storage => '저장 공간';

  @override
  String storageCount(int count) {
    return '이 기기에 문서 $count개';
  }

  @override
  String get about => '정보';

  @override
  String get version => '버전';

  @override
  String get recentDocuments => '최근';

  @override
  String get seeAll => '모두 보기';

  @override
  String get noEdits => '아직 처리본이 없습니다';
}
