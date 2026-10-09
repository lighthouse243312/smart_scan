// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for French (`fr`).
class AppLocalizationsFr extends AppLocalizations {
  AppLocalizationsFr([String locale = 'fr']) : super(locale);

  @override
  String get appTitle => 'Smart Scan';

  @override
  String get homeHeadline => 'Des documents nets,\nprêts à partager';

  @override
  String get homeSubtitle =>
      'Scannez ou choisissez une photo : l\'appli l\'accentue, retire les ombres et peut effacer l\'écriture manuscrite. Export en PDF, Word ou PNG.';

  @override
  String get scanTitle => 'Scanner un document';

  @override
  String get scanSubtitle =>
      'Avec l\'appareil photo, bords détectés et recadrés automatiquement';

  @override
  String get libraryTitle => 'Choisir dans la galerie';

  @override
  String get librarySubtitle =>
      'Photos déjà sur l\'appareil, plusieurs à la fois';

  @override
  String get pdfTitle => 'Importer un PDF';

  @override
  String get pdfSubtitle =>
      'Chaque page d\'un fichier PDF devient une page à nettoyer';

  @override
  String get exportTo => 'Exporter en';

  @override
  String get language => 'Langue';

  @override
  String get systemLanguage => 'Langue du système';

  @override
  String get stepCapture => 'Capture';

  @override
  String get stepProcess => 'Traitement';

  @override
  String get stepHandwriting => 'Manuscrit';

  @override
  String get stepExport => 'Export';

  @override
  String pageOf(int current, int total) {
    return 'Page $current/$total';
  }

  @override
  String get processTitle => 'Netteté et ombres';

  @override
  String get rotateLeft => 'Pivoter à gauche';

  @override
  String get rotateRight => 'Pivoter à droite';

  @override
  String get deletePage => 'Supprimer la page';

  @override
  String get deletePageTitle => 'Supprimer cette page ?';

  @override
  String deletePageBody(int page) {
    return 'La page $page sera retirée du document.';
  }

  @override
  String get deleteThisPage => 'Supprimer cette page';

  @override
  String get deleteAllPages => 'Supprimer toutes les pages';

  @override
  String deleteAllPagesTitle(int count) {
    return 'Supprimer les $count pages ?';
  }

  @override
  String get deleteAllPagesBody =>
      'Toutes les pages seront retirées et le document supprimé de cet appareil.';

  @override
  String get cancel => 'Annuler';

  @override
  String get delete => 'Supprimer';

  @override
  String get addScanPage => 'Scanner d\'autres pages';

  @override
  String get addLibraryPage => 'Ajouter depuis la galerie';

  @override
  String get addPdfPage => 'Ajouter depuis un fichier PDF';

  @override
  String get processing =>
      'Amélioration de la netteté et suppression des ombres…';

  @override
  String get original => 'Original';

  @override
  String get processed => 'Traité';

  @override
  String get exportNow => 'Exporter maintenant';

  @override
  String get eraseWritingShort => 'Effacer l\'écriture';

  @override
  String get handwritingTitle => 'Écriture manuscrite';

  @override
  String get hideMask => 'Masquer la zone détectée';

  @override
  String get showMask => 'Afficher la zone détectée';

  @override
  String get saveOriginal => 'Enregistrer l\'original dans Photos';

  @override
  String get methodInk => 'Couleur de l\'encre';

  @override
  String get methodAi => 'Couleur de l\'encre + IA';

  @override
  String get sensitivity => 'Sensibilité';

  @override
  String get brushView => 'Voir';

  @override
  String get brushAdd => 'Ajouter à effacer';

  @override
  String get brushKeep => 'Conserver';

  @override
  String get skip => 'Passer';

  @override
  String get eraseHandwriting => 'Effacer l\'écriture manuscrite';

  @override
  String detectingPages(int current, int total) {
    return 'Recherche de l’écriture… page $current sur $total';
  }

  @override
  String erasingPages(int current, int total) {
    return 'Effacement de l’écriture… page $current sur $total';
  }

  @override
  String get eraseAllPages => 'Effacer toutes les pages';

  @override
  String get hintNoneInk =>
      'Aucun trait de stylo d\'une autre couleur que l\'impression. Augmentez la sensibilité ; pour un stylo de la même couleur que l\'impression, utilisez « Couleur de l\'encre + IA ».';

  @override
  String get hintNoneAi =>
      'Aucune écriture manuscrite trouvée. Augmentez la sensibilité ou repassez au pinceau.';

  @override
  String hintCoverage(String percent) {
    return 'La zone rouge ($percent % de la page) sera effacée ; le jaune est du texte imprimé recouvert, qui sera reconstruit. Le pinceau permet d\'ajouter des zones ou de garder un texte marqué par erreur ; changer de méthode ou de sensibilité relance la détection.';
  }

  @override
  String get savedOriginal => 'Original enregistré dans Photos.';

  @override
  String saveFailed(String error) {
    return 'Échec de l\'enregistrement : $error';
  }

  @override
  String get exportTitle => 'Exporter le document';

  @override
  String get fileName => 'Nom du fichier';

  @override
  String get formatPdf => 'PDF';

  @override
  String get formatWord => 'Word';

  @override
  String get formatPng => 'Images PNG';

  @override
  String get formatPdfDesc => 'Un seul fichier, une page A4 par scan';

  @override
  String get formatWordDesc => 'Un fichier .docx pour Word ou Google Docs';

  @override
  String get formatPngDesc => 'Une image pleine qualité par page';

  @override
  String shareFormat(String format) {
    return 'Partager en $format';
  }

  @override
  String get saveImages => 'Enregistrer les images';

  @override
  String get newDocument => 'Nouveau document';

  @override
  String savedImages(int count) {
    return '$count images enregistrées dans la galerie';
  }

  @override
  String actionFailed(String error) {
    return 'Action impossible : $error';
  }

  @override
  String get galleryPermission =>
      'L\'accès à la photothèque n\'a pas été autorisé';

  @override
  String processingFailed(String code) {
    return 'Échec du traitement de l\'image ($code)';
  }

  @override
  String get tabScan => 'Scanner';

  @override
  String get tabDocuments => 'Documents';

  @override
  String get tabAccount => 'Compte';

  @override
  String get documentsTitle => 'Documents enregistrés';

  @override
  String get documentsEmpty => 'Aucun document pour l\'instant';

  @override
  String get documentsEmptyHint =>
      'Chaque scan conserve ici son original ; les versions traitées que vous enregistrez apparaissent à côté.';

  @override
  String pagesCount(int count) {
    return '$count pages';
  }

  @override
  String get editAgain => 'Modifier à nouveau';

  @override
  String get share => 'Partager';

  @override
  String get rename => 'Renommer';

  @override
  String get deleteDocument => 'Supprimer le document';

  @override
  String get deleteDocumentTitle => 'Supprimer ce document ?';

  @override
  String get deleteDocumentBody =>
      'Ses originaux et versions traitées seront définitivement supprimés de cet appareil.';

  @override
  String get save => 'Enregistrer';

  @override
  String get documentName => 'Nom du document';

  @override
  String get saveToLibrary => 'Enregistrer dans Documents';

  @override
  String get savedToLibrary => 'Version traitée enregistrée dans Documents';

  @override
  String get originalsKept =>
      'Les originaux sont enregistrés automatiquement dans Documents';

  @override
  String get accountGuest => 'Invité';

  @override
  String get accountGuestHint =>
      'Vos documents sont stockés sur cet appareil. Connectez-vous pour les synchroniser entre appareils (bientôt).';

  @override
  String get signIn => 'Se connecter';

  @override
  String get comingSoon => 'Bientôt';

  @override
  String get settings => 'Paramètres';

  @override
  String get storage => 'Stockage';

  @override
  String storageCount(int count) {
    return '$count documents sur cet appareil';
  }

  @override
  String get about => 'À propos';

  @override
  String get version => 'Version';

  @override
  String get recentDocuments => 'Récents';

  @override
  String get seeAll => 'Tout voir';

  @override
  String get noEdits => 'Pas encore de version traitée';
}
