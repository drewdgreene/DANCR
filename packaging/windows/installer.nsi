; DANCR Windows installer (NSIS). Cross-builds on Linux/macOS with makensis, or runs on Windows.
;
;   makensis -DAPPDIR=dist/DANCR -DVERSION=2.0.0 -DOUTFILE=dist/DANCR-Setup-2.0.0.exe \
;            -DICON=packaging/icon.ico packaging/windows/installer.nsi
;
; APPDIR is the one-folder PyInstaller output (DANCR.exe, dancr-cli.exe, _internal/...).

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!ifndef APPDIR
  !define APPDIR "dist\DANCR"
!endif
!ifndef OUTFILE
  !define OUTFILE "dist\DANCR-Setup-${VERSION}.exe"
!endif
!ifndef ICON
  !define ICON "${__FILEDIR__}/../icon.ico"      ; packaging/icon.ico next to this script's parent
!endif

Unicode true
SetCompressor /SOLID lzma
Name "DANCR"
Caption "DANCR ${VERSION} Setup"
BrandingText "DANCR"
OutFile "${OUTFILE}"
RequestExecutionLevel user                       ; per-user install, no UAC prompt
InstallDir "$LOCALAPPDATA\Programs\DANCR"
InstallDirRegKey HKCU "Software\DANCR" "InstallDir"
ShowInstDetails show
ShowUninstDetails show
VIProductVersion "2.0.0.0"
VIAddVersionKey "ProductName" "DANCR"
VIAddVersionKey "FileDescription" "DANCR Setup"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "Copyright (c) 2026 Drew Greene"

!include "MUI2.nsh"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\DANCR.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Run DANCR"
!define MUI_ICON "${ICON}"
!define MUI_UNICON "${ICON}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

LangString DESC_App ${LANG_ENGLISH} "The DANCR application, and the command line and MCP tools (dancr-cli)."
LangString DESC_StartMenu ${LANG_ENGLISH} "Add DANCR to the Start Menu."
LangString DESC_Desktop ${LANG_ENGLISH} "Add a DANCR shortcut to the desktop."

Section "DANCR" SEC_APP
  SectionIn RO
  SetRegView 64
  SetOutPath "$INSTDIR"
  File /r "${APPDIR}\*.*"

  WriteRegStr HKCU "Software\DANCR" "InstallDir" "$INSTDIR"

  ; Add or Remove Programs entry
  !define UNINST "Software\Microsoft\Windows\CurrentVersion\Uninstall\DANCR"
  WriteRegStr   HKCU "${UNINST}" "DisplayName"     "DANCR"
  WriteRegStr   HKCU "${UNINST}" "DisplayVersion"  "${VERSION}"
  WriteRegStr   HKCU "${UNINST}" "Publisher"       "Drew Greene"
  WriteRegStr   HKCU "${UNINST}" "DisplayIcon"     "$INSTDIR\DANCR.exe"
  WriteRegStr   HKCU "${UNINST}" "InstallLocation" "$INSTDIR"
  WriteRegStr   HKCU "${UNINST}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr   HKCU "${UNINST}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
  WriteRegDWORD HKCU "${UNINST}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST}" "NoRepair" 1
  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

Section "Start Menu shortcut" SEC_START
  CreateDirectory "$SMPROGRAMS\DANCR"
  CreateShortcut "$SMPROGRAMS\DANCR\DANCR.lnk" "$INSTDIR\DANCR.exe"
  CreateShortcut "$SMPROGRAMS\DANCR\Uninstall DANCR.lnk" "$INSTDIR\uninstall.exe"
SectionEnd

Section /o "Desktop shortcut" SEC_DESKTOP
  CreateShortcut "$DESKTOP\DANCR.lnk" "$INSTDIR\DANCR.exe"
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_APP} $(DESC_App)
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_START} $(DESC_StartMenu)
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_DESKTOP} $(DESC_Desktop)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "Uninstall"
  SetRegView 64
  Delete "$SMPROGRAMS\DANCR\DANCR.lnk"
  Delete "$SMPROGRAMS\DANCR\Uninstall DANCR.lnk"
  RMDir "$SMPROGRAMS\DANCR"
  Delete "$DESKTOP\DANCR.lnk"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DANCR"
  DeleteRegKey HKCU "Software\DANCR"
  RMDir /r "$INSTDIR"
SectionEnd
