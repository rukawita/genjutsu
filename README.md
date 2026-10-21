# Genjutsu-Genjutsu (Open Genjutsu Studio)

Estudio local de **Motion Transfer** (`motion-control`) y **Objects Swap** (`replace-objects`) reconstruido mediante ingeniería inversa con [REA (`morluto/rea`)](https://github.com/morluto/rea) e impulsado por el SDK oficial **`@google/genai`** (**Gemini Omni 1.1 Flash** & **Veo 3.1**).

---

## Características Principales

- **2 Modos de Manipulación de Video**:
  - **Motion Transfer (`motion-control`)**: Extrae el movimiento y el trabajo de cámara de un video de referencia (`@video1`) y lo transfiere a tus personajes u objetos (`@image1`, `@image2`...).
  - **Objects Swap (`replace-objects`)**: Reemplaza sujetos, vestimenta u objetos en un video existente preservando el resto de la escena (`Keep everything else the same`).
- **Control Avanzado de Escena (`scene_mode`)**:
  - `video`: Conserva el fondo del video de movimiento (`@video1`).
  - `character`: Usa el fondo de la primera imagen de referencia (`@image1`).
  - `location`: Permite subir una imagen dedicada para el escenario (`@location`).
- **Traducción Automática a `gemini-omni-1.1-flash` (Interactions API & Files API)**:
  - Sube automáticamente tus videos e imágenes locales a la **Files API** de Google (`ai.files.upload`).
  - Traduce las menciones `@video1`, `@image1`, `@location` a las etiquetas multimodales oficiales `[# Sources <VIDEO_0>@Video1]` y `[# References <IMAGE_REF_0>@Image1 <VIDEO_REF_0>@Video1]`.
- **Soporte para `veo-3.1-fast-generate-preview` y `veo-3.1-generate-preview`**.

---

## Instalación y Uso Rápido

### 1. Clonar e instalar dependencias

```bash
git clone https://github.com/rukawita/genjutsu.git
cd genjutsu
npm install
```

### 2. Iniciar el servidor local

```bash
npm start
```

Abre **`http://127.0.0.1:3456`** en tu navegador.

### 3. Configurar tu API Key

Puedes ingresar tu API Key de [Google AI Studio](https://aistudio.google.com/apikey) directamente en la barra lateral de la interfaz web, o definirla como variable de entorno antes de iniciar:

```bash
# En Windows (PowerShell)
$env:GEMINI_API_KEY="AIzaSy..."
npm start
```

---

## Estructura del Proyecto

- `server.mjs`: Servidor HTTP Node.js que gestiona la subida a Google Files API, el armado de interacciones multimodales para `gemini-omni-1.1-flash` y `veo-3.1`, y la descarga automática de los videos generados.
- `index.html`: Interfaz web interactiva con soporte de etiquetas `@video1`, `@image1`, `@location`, previsualización en vivo de payloads JSON y reproductor de video integrado.
