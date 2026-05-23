#------------------------------------------------------------------------------
# AppImage Configuration
# Builds a portable AppImage for IOWarp Core
#------------------------------------------------------------------------------

# Define AppImage staging directory
set(APPIMAGE_STAGING_DIR "${CMAKE_BINARY_DIR}/AppDir")

#------------------------------------------------------------------------------
# Target: appimage-stage
# Stages the installation to AppDir structure for AppImage
#------------------------------------------------------------------------------
add_custom_target(appimage-stage
    COMMENT "Staging AppImage directory: ${APPIMAGE_STAGING_DIR}"
    COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --prefix ${APPIMAGE_STAGING_DIR}/usr
    COMMAND ${CMAKE_COMMAND} -E make_directory ${APPIMAGE_STAGING_DIR}/usr/share/applications
    COMMAND ${CMAKE_COMMAND} -E make_directory ${APPIMAGE_STAGING_DIR}/usr/share/icons/hicolor/256x256/apps
    COMMAND ${CMAKE_COMMAND} -E echo "[Desktop Entry]" > ${APPIMAGE_STAGING_DIR}/usr/share/applications/clio_run.desktop
    COMMAND ${CMAKE_COMMAND} -E echo "Type=Application" >> ${APPIMAGE_STAGING_DIR}/usr/share/applications/clio_run.desktop
    COMMAND ${CMAKE_COMMAND} -E echo "Name=IOWarp Core Runtime" >> ${APPIMAGE_STAGING_DIR}/usr/share/applications/clio_run.desktop
    COMMAND ${CMAKE_COMMAND} -E echo "Exec=clio_run" >> ${APPIMAGE_STAGING_DIR}/usr/share/applications/clio_run.desktop
    COMMAND ${CMAKE_COMMAND} -E echo "Icon=clio_run" >> ${APPIMAGE_STAGING_DIR}/usr/share/applications/clio_run.desktop
    COMMAND ${CMAKE_COMMAND} -E echo "Categories=System;" >> ${APPIMAGE_STAGING_DIR}/usr/share/applications/clio_run.desktop
    COMMAND ${CMAKE_COMMAND} -E echo "Terminal=true" >> ${APPIMAGE_STAGING_DIR}/usr/share/applications/clio_run.desktop
    COMMAND touch ${APPIMAGE_STAGING_DIR}/usr/share/icons/hicolor/256x256/apps/clio_run.png
    COMMAND ${CMAKE_COMMAND} -E echo "# Placeholder icon for AppImage" > ${APPIMAGE_STAGING_DIR}/usr/share/icons/hicolor/256x256/apps/clio_run.png
)

#------------------------------------------------------------------------------
# Target: appimage
# Builds the final AppImage using appimagetool
#------------------------------------------------------------------------------
add_custom_target(appimage
    COMMENT "Building AppImage: ${CMAKE_BINARY_DIR}/iowarp-core-${PROJECT_VERSION}-x86_64.AppImage"
    DEPENDS appimage-stage
    COMMAND bash -c "
        set -e

        # Find or download appimagetool
        APPIMAGETOOL=''

        # Check if appimagetool exists in current directory or build directory
        if [ -x './appimagetool' ]; then
            APPIMAGETOOL='./appimagetool'
        elif [ -x '${CMAKE_BINARY_DIR}/appimagetool' ]; then
            APPIMAGETOOL='${CMAKE_BINARY_DIR}/appimagetool'
        else
            # Try to find it in PATH
            if command -v appimagetool &>/dev/null; then
                APPIMAGETOOL='appimagetool'
            else
                # Download appimagetool
                echo 'appimagetool not found, downloading from GitHub...'
                cd '${CMAKE_BINARY_DIR}'
                wget -q -O appimagetool-x86_64.AppImage 'https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage'
                chmod +x appimagetool-x86_64.AppImage
                APPIMAGETOOL='${CMAKE_BINARY_DIR}/appimagetool-x86_64.AppImage'
            fi
        fi

        # Build the AppImage
        export ARCH=x86_64
        \$APPIMAGETOOL '${APPIMAGE_STAGING_DIR}' '${CMAKE_BINARY_DIR}/iowarp-core-${PROJECT_VERSION}-x86_64.AppImage'
    "
)

#------------------------------------------------------------------------------
# AppRun script for AppImage
# Creates the AppRun entry point that executes clio_run from within the AppImage
#------------------------------------------------------------------------------
file(WRITE "${APPIMAGE_STAGING_DIR}/AppRun" "#!/bin/bash
# AppRun script for IOWarp Core AppImage
exec \"\$APPDIR/usr/bin/clio_run\" \"\$@\"
")

# Make AppRun executable after writing
file(INSTALL "${APPIMAGE_STAGING_DIR}/AppRun"
    DESTINATION "${APPIMAGE_STAGING_DIR}"
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

message(STATUS "AppImage target registered — run: cmake --build . --target appimage")
