import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:intl/intl.dart' as intl;

import 'app_localizations_en.dart';
import 'app_localizations_fr.dart';
import 'app_localizations_ja.dart';
import 'app_localizations_ko.dart';
import 'app_localizations_vi.dart';
import 'app_localizations_zh.dart';

// ignore_for_file: type=lint

/// Callers can lookup localized strings with an instance of AppLocalizations
/// returned by `AppLocalizations.of(context)`.
///
/// Applications need to include `AppLocalizations.delegate()` in their app's
/// `localizationDelegates` list, and the locales they support in the app's
/// `supportedLocales` list. For example:
///
/// ```dart
/// import 'l10n/app_localizations.dart';
///
/// return MaterialApp(
///   localizationsDelegates: AppLocalizations.localizationsDelegates,
///   supportedLocales: AppLocalizations.supportedLocales,
///   home: MyApplicationHome(),
/// );
/// ```
///
/// ## Update pubspec.yaml
///
/// Please make sure to update your pubspec.yaml to include the following
/// packages:
///
/// ```yaml
/// dependencies:
///   # Internationalization support.
///   flutter_localizations:
///     sdk: flutter
///   intl: any # Use the pinned version from flutter_localizations
///
///   # Rest of dependencies
/// ```
///
/// ## iOS Applications
///
/// iOS applications define key application metadata, including supported
/// locales, in an Info.plist file that is built into the application bundle.
/// To configure the locales supported by your app, you’ll need to edit this
/// file.
///
/// First, open your project’s ios/Runner.xcworkspace Xcode workspace file.
/// Then, in the Project Navigator, open the Info.plist file under the Runner
/// project’s Runner folder.
///
/// Next, select the Information Property List item, select Add Item from the
/// Editor menu, then select Localizations from the pop-up menu.
///
/// Select and expand the newly-created Localizations item then, for each
/// locale your application supports, add a new item and select the locale
/// you wish to add from the pop-up menu in the Value field. This list should
/// be consistent with the languages listed in the AppLocalizations.supportedLocales
/// property.
abstract class AppLocalizations {
  AppLocalizations(String locale)
    : localeName = intl.Intl.canonicalizedLocale(locale.toString());

  final String localeName;

  static AppLocalizations of(BuildContext context) {
    return Localizations.of<AppLocalizations>(context, AppLocalizations)!;
  }

  static const LocalizationsDelegate<AppLocalizations> delegate =
      _AppLocalizationsDelegate();

  /// A list of this localizations delegate along with the default localizations
  /// delegates.
  ///
  /// Returns a list of localizations delegates containing this delegate along with
  /// GlobalMaterialLocalizations.delegate, GlobalCupertinoLocalizations.delegate,
  /// and GlobalWidgetsLocalizations.delegate.
  ///
  /// Additional delegates can be added by appending to this list in
  /// MaterialApp. This list does not have to be used at all if a custom list
  /// of delegates is preferred or required.
  static const List<LocalizationsDelegate<dynamic>> localizationsDelegates =
      <LocalizationsDelegate<dynamic>>[
        delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
      ];

  /// A list of this localizations delegate's supported locales.
  static const List<Locale> supportedLocales = <Locale>[
    Locale('en'),
    Locale('fr'),
    Locale('ja'),
    Locale('ko'),
    Locale('vi'),
    Locale('zh'),
  ];

  /// No description provided for @appTitle.
  ///
  /// In en, this message translates to:
  /// **'Smart Scan'**
  String get appTitle;

  /// No description provided for @homeHeadline.
  ///
  /// In en, this message translates to:
  /// **'Tidy documents,\nready to share'**
  String get homeHeadline;

  /// No description provided for @homeSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Scan or pick a photo: the app sharpens it, removes shadows and can erase handwriting. Export to PDF, Word or PNG.'**
  String get homeSubtitle;

  /// No description provided for @scanTitle.
  ///
  /// In en, this message translates to:
  /// **'Scan document'**
  String get scanTitle;

  /// No description provided for @scanSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Use the camera; edges are found and cropped automatically'**
  String get scanSubtitle;

  /// No description provided for @libraryTitle.
  ///
  /// In en, this message translates to:
  /// **'Pick from library'**
  String get libraryTitle;

  /// No description provided for @librarySubtitle.
  ///
  /// In en, this message translates to:
  /// **'Use photos already on this device, several at once'**
  String get librarySubtitle;

  /// No description provided for @pdfTitle.
  ///
  /// In en, this message translates to:
  /// **'Import a PDF'**
  String get pdfTitle;

  /// No description provided for @pdfSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Each page of a PDF file becomes a page to clean'**
  String get pdfSubtitle;

  /// No description provided for @exportTo.
  ///
  /// In en, this message translates to:
  /// **'Export to'**
  String get exportTo;

  /// No description provided for @language.
  ///
  /// In en, this message translates to:
  /// **'Language'**
  String get language;

  /// No description provided for @systemLanguage.
  ///
  /// In en, this message translates to:
  /// **'System default'**
  String get systemLanguage;

  /// No description provided for @stepCapture.
  ///
  /// In en, this message translates to:
  /// **'Capture'**
  String get stepCapture;

  /// No description provided for @stepProcess.
  ///
  /// In en, this message translates to:
  /// **'Process'**
  String get stepProcess;

  /// No description provided for @stepHandwriting.
  ///
  /// In en, this message translates to:
  /// **'Handwriting'**
  String get stepHandwriting;

  /// No description provided for @stepExport.
  ///
  /// In en, this message translates to:
  /// **'Export'**
  String get stepExport;

  /// No description provided for @pageOf.
  ///
  /// In en, this message translates to:
  /// **'Page {current}/{total}'**
  String pageOf(int current, int total);

  /// No description provided for @processTitle.
  ///
  /// In en, this message translates to:
  /// **'Sharpen & clean'**
  String get processTitle;

  /// No description provided for @rotateLeft.
  ///
  /// In en, this message translates to:
  /// **'Rotate left'**
  String get rotateLeft;

  /// No description provided for @rotateRight.
  ///
  /// In en, this message translates to:
  /// **'Rotate right'**
  String get rotateRight;

  /// No description provided for @deletePage.
  ///
  /// In en, this message translates to:
  /// **'Delete page'**
  String get deletePage;

  /// No description provided for @deletePageTitle.
  ///
  /// In en, this message translates to:
  /// **'Delete this page?'**
  String get deletePageTitle;

  /// No description provided for @deletePageBody.
  ///
  /// In en, this message translates to:
  /// **'Page {page} will be removed from the document.'**
  String deletePageBody(int page);

  /// No description provided for @deleteThisPage.
  ///
  /// In en, this message translates to:
  /// **'Delete this page'**
  String get deleteThisPage;

  /// No description provided for @deleteAllPages.
  ///
  /// In en, this message translates to:
  /// **'Delete all pages'**
  String get deleteAllPages;

  /// No description provided for @deleteAllPagesTitle.
  ///
  /// In en, this message translates to:
  /// **'Delete all {count} pages?'**
  String deleteAllPagesTitle(int count);

  /// No description provided for @deleteAllPagesBody.
  ///
  /// In en, this message translates to:
  /// **'All pages will be removed and the document deleted from this device.'**
  String get deleteAllPagesBody;

  /// No description provided for @cancel.
  ///
  /// In en, this message translates to:
  /// **'Cancel'**
  String get cancel;

  /// No description provided for @delete.
  ///
  /// In en, this message translates to:
  /// **'Delete'**
  String get delete;

  /// No description provided for @addScanPage.
  ///
  /// In en, this message translates to:
  /// **'Scan more pages'**
  String get addScanPage;

  /// No description provided for @addLibraryPage.
  ///
  /// In en, this message translates to:
  /// **'Add from library'**
  String get addLibraryPage;

  /// No description provided for @addPdfPage.
  ///
  /// In en, this message translates to:
  /// **'Add from a PDF file'**
  String get addPdfPage;

  /// No description provided for @processing.
  ///
  /// In en, this message translates to:
  /// **'Sharpening and removing shadows…'**
  String get processing;

  /// No description provided for @original.
  ///
  /// In en, this message translates to:
  /// **'Original'**
  String get original;

  /// No description provided for @processed.
  ///
  /// In en, this message translates to:
  /// **'Processed'**
  String get processed;

  /// No description provided for @exportNow.
  ///
  /// In en, this message translates to:
  /// **'Export now'**
  String get exportNow;

  /// No description provided for @eraseWritingShort.
  ///
  /// In en, this message translates to:
  /// **'Erase writing'**
  String get eraseWritingShort;

  /// No description provided for @handwritingTitle.
  ///
  /// In en, this message translates to:
  /// **'Handwriting'**
  String get handwritingTitle;

  /// No description provided for @hideMask.
  ///
  /// In en, this message translates to:
  /// **'Hide detected area'**
  String get hideMask;

  /// No description provided for @showMask.
  ///
  /// In en, this message translates to:
  /// **'Show detected area'**
  String get showMask;

  /// No description provided for @saveOriginal.
  ///
  /// In en, this message translates to:
  /// **'Save original to Photos'**
  String get saveOriginal;

  /// No description provided for @methodInk.
  ///
  /// In en, this message translates to:
  /// **'Ink colour'**
  String get methodInk;

  /// No description provided for @methodAi.
  ///
  /// In en, this message translates to:
  /// **'Ink colour + AI'**
  String get methodAi;

  /// No description provided for @sensitivity.
  ///
  /// In en, this message translates to:
  /// **'Sensitivity'**
  String get sensitivity;

  /// No description provided for @brushView.
  ///
  /// In en, this message translates to:
  /// **'View'**
  String get brushView;

  /// No description provided for @brushAdd.
  ///
  /// In en, this message translates to:
  /// **'Mark to erase'**
  String get brushAdd;

  /// No description provided for @brushKeep.
  ///
  /// In en, this message translates to:
  /// **'Keep'**
  String get brushKeep;

  /// No description provided for @skip.
  ///
  /// In en, this message translates to:
  /// **'Skip'**
  String get skip;

  /// No description provided for @eraseHandwriting.
  ///
  /// In en, this message translates to:
  /// **'Erase handwriting'**
  String get eraseHandwriting;

  /// No description provided for @detectingPages.
  ///
  /// In en, this message translates to:
  /// **'Finding handwriting… page {current} of {total}'**
  String detectingPages(int current, int total);

  /// No description provided for @erasingPages.
  ///
  /// In en, this message translates to:
  /// **'Erasing handwriting… page {current} of {total}'**
  String erasingPages(int current, int total);

  /// No description provided for @eraseAllPages.
  ///
  /// In en, this message translates to:
  /// **'Erase all pages'**
  String get eraseAllPages;

  /// No description provided for @hintNoneInk.
  ///
  /// In en, this message translates to:
  /// **'No pen strokes in a colour different from the print. Try a higher sensitivity; for pen the same colour as the print, use \"Ink colour + AI\".'**
  String get hintNoneInk;

  /// No description provided for @hintNoneAi.
  ///
  /// In en, this message translates to:
  /// **'No handwriting found. Try a higher sensitivity or paint over it with the brush.'**
  String get hintNoneAi;

  /// No description provided for @hintCoverage.
  ///
  /// In en, this message translates to:
  /// **'The red area ({percent}% of the page) will be erased; yellow is print written over, which will be rebuilt. Use the brush to add areas or keep text marked by mistake; changing the method or sensitivity detects again from scratch.'**
  String hintCoverage(String percent);

  /// No description provided for @savedOriginal.
  ///
  /// In en, this message translates to:
  /// **'Original saved to Photos.'**
  String get savedOriginal;

  /// No description provided for @saveFailed.
  ///
  /// In en, this message translates to:
  /// **'Save failed: {error}'**
  String saveFailed(String error);

  /// No description provided for @exportTitle.
  ///
  /// In en, this message translates to:
  /// **'Export document'**
  String get exportTitle;

  /// No description provided for @fileName.
  ///
  /// In en, this message translates to:
  /// **'File name'**
  String get fileName;

  /// No description provided for @formatPdf.
  ///
  /// In en, this message translates to:
  /// **'PDF'**
  String get formatPdf;

  /// No description provided for @formatWord.
  ///
  /// In en, this message translates to:
  /// **'Word'**
  String get formatWord;

  /// No description provided for @formatPng.
  ///
  /// In en, this message translates to:
  /// **'PNG images'**
  String get formatPng;

  /// No description provided for @formatPdfDesc.
  ///
  /// In en, this message translates to:
  /// **'One file, one A4 page per scan'**
  String get formatPdfDesc;

  /// No description provided for @formatWordDesc.
  ///
  /// In en, this message translates to:
  /// **'A .docx file for Word or Google Docs'**
  String get formatWordDesc;

  /// No description provided for @formatPngDesc.
  ///
  /// In en, this message translates to:
  /// **'One full-quality image per page'**
  String get formatPngDesc;

  /// No description provided for @shareFormat.
  ///
  /// In en, this message translates to:
  /// **'Share {format}'**
  String shareFormat(String format);

  /// No description provided for @saveImages.
  ///
  /// In en, this message translates to:
  /// **'Save images'**
  String get saveImages;

  /// No description provided for @newDocument.
  ///
  /// In en, this message translates to:
  /// **'New document'**
  String get newDocument;

  /// No description provided for @savedImages.
  ///
  /// In en, this message translates to:
  /// **'Saved {count} images to the gallery'**
  String savedImages(int count);

  /// No description provided for @actionFailed.
  ///
  /// In en, this message translates to:
  /// **'Couldn\'t do that: {error}'**
  String actionFailed(String error);

  /// No description provided for @galleryPermission.
  ///
  /// In en, this message translates to:
  /// **'Photo library access was not granted'**
  String get galleryPermission;

  /// No description provided for @processingFailed.
  ///
  /// In en, this message translates to:
  /// **'Image processing failed ({code})'**
  String processingFailed(String code);

  /// No description provided for @tabScan.
  ///
  /// In en, this message translates to:
  /// **'Scan'**
  String get tabScan;

  /// No description provided for @tabDocuments.
  ///
  /// In en, this message translates to:
  /// **'Documents'**
  String get tabDocuments;

  /// No description provided for @tabAccount.
  ///
  /// In en, this message translates to:
  /// **'Account'**
  String get tabAccount;

  /// No description provided for @documentsTitle.
  ///
  /// In en, this message translates to:
  /// **'Saved documents'**
  String get documentsTitle;

  /// No description provided for @documentsEmpty.
  ///
  /// In en, this message translates to:
  /// **'No documents yet'**
  String get documentsEmpty;

  /// No description provided for @documentsEmptyHint.
  ///
  /// In en, this message translates to:
  /// **'Every scan keeps its original here; processed versions you save appear next to it.'**
  String get documentsEmptyHint;

  /// No description provided for @pagesCount.
  ///
  /// In en, this message translates to:
  /// **'{count} pages'**
  String pagesCount(int count);

  /// No description provided for @editAgain.
  ///
  /// In en, this message translates to:
  /// **'Edit again'**
  String get editAgain;

  /// No description provided for @share.
  ///
  /// In en, this message translates to:
  /// **'Share'**
  String get share;

  /// No description provided for @rename.
  ///
  /// In en, this message translates to:
  /// **'Rename'**
  String get rename;

  /// No description provided for @deleteDocument.
  ///
  /// In en, this message translates to:
  /// **'Delete document'**
  String get deleteDocument;

  /// No description provided for @deleteDocumentTitle.
  ///
  /// In en, this message translates to:
  /// **'Delete this document?'**
  String get deleteDocumentTitle;

  /// No description provided for @deleteDocumentBody.
  ///
  /// In en, this message translates to:
  /// **'Its originals and processed versions will be removed from this device for good.'**
  String get deleteDocumentBody;

  /// No description provided for @save.
  ///
  /// In en, this message translates to:
  /// **'Save'**
  String get save;

  /// No description provided for @documentName.
  ///
  /// In en, this message translates to:
  /// **'Document name'**
  String get documentName;

  /// No description provided for @saveToLibrary.
  ///
  /// In en, this message translates to:
  /// **'Save to documents'**
  String get saveToLibrary;

  /// No description provided for @savedToLibrary.
  ///
  /// In en, this message translates to:
  /// **'Processed version saved to documents'**
  String get savedToLibrary;

  /// No description provided for @originalsKept.
  ///
  /// In en, this message translates to:
  /// **'Originals are saved automatically in Documents'**
  String get originalsKept;

  /// No description provided for @accountGuest.
  ///
  /// In en, this message translates to:
  /// **'Guest'**
  String get accountGuest;

  /// No description provided for @accountGuestHint.
  ///
  /// In en, this message translates to:
  /// **'Your documents are stored on this device. Sign in to sync across devices (coming soon).'**
  String get accountGuestHint;

  /// No description provided for @signIn.
  ///
  /// In en, this message translates to:
  /// **'Sign in'**
  String get signIn;

  /// No description provided for @comingSoon.
  ///
  /// In en, this message translates to:
  /// **'Coming soon'**
  String get comingSoon;

  /// No description provided for @settings.
  ///
  /// In en, this message translates to:
  /// **'Settings'**
  String get settings;

  /// No description provided for @storage.
  ///
  /// In en, this message translates to:
  /// **'Storage'**
  String get storage;

  /// No description provided for @storageCount.
  ///
  /// In en, this message translates to:
  /// **'{count} documents on this device'**
  String storageCount(int count);

  /// No description provided for @about.
  ///
  /// In en, this message translates to:
  /// **'About'**
  String get about;

  /// No description provided for @version.
  ///
  /// In en, this message translates to:
  /// **'Version'**
  String get version;

  /// No description provided for @recentDocuments.
  ///
  /// In en, this message translates to:
  /// **'Recent'**
  String get recentDocuments;

  /// No description provided for @seeAll.
  ///
  /// In en, this message translates to:
  /// **'See all'**
  String get seeAll;

  /// No description provided for @noEdits.
  ///
  /// In en, this message translates to:
  /// **'No processed version yet'**
  String get noEdits;
}

class _AppLocalizationsDelegate
    extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  Future<AppLocalizations> load(Locale locale) {
    return SynchronousFuture<AppLocalizations>(lookupAppLocalizations(locale));
  }

  @override
  bool isSupported(Locale locale) => <String>[
    'en',
    'fr',
    'ja',
    'ko',
    'vi',
    'zh',
  ].contains(locale.languageCode);

  @override
  bool shouldReload(_AppLocalizationsDelegate old) => false;
}

AppLocalizations lookupAppLocalizations(Locale locale) {
  // Lookup logic when only language code is specified.
  switch (locale.languageCode) {
    case 'en':
      return AppLocalizationsEn();
    case 'fr':
      return AppLocalizationsFr();
    case 'ja':
      return AppLocalizationsJa();
    case 'ko':
      return AppLocalizationsKo();
    case 'vi':
      return AppLocalizationsVi();
    case 'zh':
      return AppLocalizationsZh();
  }

  throw FlutterError(
    'AppLocalizations.delegate failed to load unsupported locale "$locale". This is likely '
    'an issue with the localizations generation tool. Please file an issue '
    'on GitHub with a reproducible sample app and the gen-l10n configuration '
    'that was used.',
  );
}
