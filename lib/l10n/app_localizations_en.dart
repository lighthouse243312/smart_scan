// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for English (`en`).
class AppLocalizationsEn extends AppLocalizations {
  AppLocalizationsEn([String locale = 'en']) : super(locale);

  @override
  String get appTitle => 'Smart Scan';

  @override
  String get homeHeadline => 'Tidy documents,\nready to share';

  @override
  String get homeSubtitle =>
      'Scan or pick a photo: the app sharpens it, removes shadows and can erase handwriting. Export to PDF, Word or PNG.';

  @override
  String get scanTitle => 'Scan document';

  @override
  String get scanSubtitle =>
      'Use the camera; edges are found and cropped automatically';

  @override
  String get libraryTitle => 'Pick from library';

  @override
  String get librarySubtitle =>
      'Use photos already on this device, several at once';

  @override
  String get pdfTitle => 'Import a PDF';

  @override
  String get pdfSubtitle => 'Each page of a PDF file becomes a page to clean';

  @override
  String get exportTo => 'Export to';

  @override
  String get language => 'Language';

  @override
  String get systemLanguage => 'System default';

  @override
  String get stepCapture => 'Capture';

  @override
  String get stepProcess => 'Process';

  @override
  String get stepHandwriting => 'Handwriting';

  @override
  String get stepExport => 'Export';

  @override
  String pageOf(int current, int total) {
    return 'Page $current/$total';
  }

  @override
  String get processTitle => 'Sharpen & clean';

  @override
  String get rotateLeft => 'Rotate left';

  @override
  String get rotateRight => 'Rotate right';

  @override
  String get deletePage => 'Delete page';

  @override
  String get deletePageTitle => 'Delete this page?';

  @override
  String deletePageBody(int page) {
    return 'Page $page will be removed from the document.';
  }

  @override
  String get deleteThisPage => 'Delete this page';

  @override
  String get deleteAllPages => 'Delete all pages';

  @override
  String deleteAllPagesTitle(int count) {
    return 'Delete all $count pages?';
  }

  @override
  String get deleteAllPagesBody =>
      'All pages will be removed and the document deleted from this device.';

  @override
  String get cancel => 'Cancel';

  @override
  String get delete => 'Delete';

  @override
  String get addScanPage => 'Scan more pages';

  @override
  String get addLibraryPage => 'Add from library';

  @override
  String get addPdfPage => 'Add from a PDF file';

  @override
  String get processing => 'Sharpening and removing shadows…';

  @override
  String get original => 'Original';

  @override
  String get processed => 'Processed';

  @override
  String get exportNow => 'Export now';

  @override
  String get eraseWritingShort => 'Erase writing';

  @override
  String get handwritingTitle => 'Handwriting';

  @override
  String get hideMask => 'Hide detected area';

  @override
  String get showMask => 'Show detected area';

  @override
  String get saveOriginal => 'Save original to Photos';

  @override
  String get methodInk => 'Ink colour';

  @override
  String get methodAi => 'Ink colour + AI';

  @override
  String get sensitivity => 'Sensitivity';

  @override
  String get brushView => 'View';

  @override
  String get brushAdd => 'Mark to erase';

  @override
  String get brushKeep => 'Keep';

  @override
  String get skip => 'Skip';

  @override
  String get eraseHandwriting => 'Erase handwriting';

  @override
  String detectingPages(int current, int total) {
    return 'Finding handwriting… page $current of $total';
  }

  @override
  String erasingPages(int current, int total) {
    return 'Erasing handwriting… page $current of $total';
  }

  @override
  String get eraseAllPages => 'Erase all pages';

  @override
  String get hintNoneInk =>
      'No pen strokes in a colour different from the print. Try a higher sensitivity; for pen the same colour as the print, use \"Ink colour + AI\".';

  @override
  String get hintNoneAi =>
      'No handwriting found. Try a higher sensitivity or paint over it with the brush.';

  @override
  String hintCoverage(String percent) {
    return 'The red area ($percent% of the page) will be erased; yellow is print written over, which will be rebuilt. Use the brush to add areas or keep text marked by mistake; changing the method or sensitivity detects again from scratch.';
  }

  @override
  String get savedOriginal => 'Original saved to Photos.';

  @override
  String saveFailed(String error) {
    return 'Save failed: $error';
  }

  @override
  String get exportTitle => 'Export document';

  @override
  String get fileName => 'File name';

  @override
  String get formatPdf => 'PDF';

  @override
  String get formatWord => 'Word';

  @override
  String get formatPng => 'PNG images';

  @override
  String get formatPdfDesc => 'One file, one A4 page per scan';

  @override
  String get formatWordDesc => 'A .docx file for Word or Google Docs';

  @override
  String get formatPngDesc => 'One full-quality image per page';

  @override
  String shareFormat(String format) {
    return 'Share $format';
  }

  @override
  String get saveImages => 'Save images';

  @override
  String get newDocument => 'New document';

  @override
  String savedImages(int count) {
    return 'Saved $count images to the gallery';
  }

  @override
  String actionFailed(String error) {
    return 'Couldn\'t do that: $error';
  }

  @override
  String get galleryPermission => 'Photo library access was not granted';

  @override
  String processingFailed(String code) {
    return 'Image processing failed ($code)';
  }

  @override
  String get tabScan => 'Scan';

  @override
  String get tabDocuments => 'Documents';

  @override
  String get tabAccount => 'Account';

  @override
  String get documentsTitle => 'Saved documents';

  @override
  String get documentsEmpty => 'No documents yet';

  @override
  String get documentsEmptyHint =>
      'Every scan keeps its original here; processed versions you save appear next to it.';

  @override
  String pagesCount(int count) {
    return '$count pages';
  }

  @override
  String get editAgain => 'Edit again';

  @override
  String get share => 'Share';

  @override
  String get rename => 'Rename';

  @override
  String get deleteDocument => 'Delete document';

  @override
  String get deleteDocumentTitle => 'Delete this document?';

  @override
  String get deleteDocumentBody =>
      'Its originals and processed versions will be removed from this device for good.';

  @override
  String get save => 'Save';

  @override
  String get documentName => 'Document name';

  @override
  String get saveToLibrary => 'Save to documents';

  @override
  String get savedToLibrary => 'Processed version saved to documents';

  @override
  String get originalsKept => 'Originals are saved automatically in Documents';

  @override
  String get accountGuest => 'Guest';

  @override
  String get accountGuestHint =>
      'Your documents are stored on this device. Sign in to sync across devices (coming soon).';

  @override
  String get signIn => 'Sign in';

  @override
  String get comingSoon => 'Coming soon';

  @override
  String get settings => 'Settings';

  @override
  String get storage => 'Storage';

  @override
  String storageCount(int count) {
    return '$count documents on this device';
  }

  @override
  String get about => 'About';

  @override
  String get version => 'Version';

  @override
  String get recentDocuments => 'Recent';

  @override
  String get seeAll => 'See all';

  @override
  String get noEdits => 'No processed version yet';
}
