import flet as ft
# MODIFICADO: Importar predict_age_real en lugar de mock_predict_age
from logic import predict_age_real, get_recommendations, PLACEHOLDER_IMAGE_URL, PLACEHOLDER_IMAGE_UPLOADED

def main(page: ft.Page):
    page.title = "Diagnóstico de Envejecimiento Biológico"
    page.vertical_alignment = ft.CrossAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.ADAPTIVE
    page.theme_mode = ft.ThemeMode.LIGHT
    
    # Configuración de fuente y estilo
    page.fonts = {
        "Inter": "https://fonts.gstatic.com/s/inter/v13/UuauNSZRl9HvhSZTv3C_A6I.ttf"
    }
    page.theme = ft.Theme(font_family="Inter")

    # Referencias para los componentes de la interfaz de usuario
    txt_age = ft.TextField(
        label="Edad Cronológica Actual",
        keyboard_type=ft.KeyboardType.NUMBER,
        width=250,
        height=50,
        border_radius=10,
        max_length=3,
        input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9]", replacement_string="")
    )
    
    img_display = ft.Image(
        src=PLACEHOLDER_IMAGE_URL,
        width=250,
        height=250,
        fit=ft.ImageFit.COVER,
        border_radius=ft.border_radius.all(15),
        tooltip="Rostro a analizar"
    )
    
    results_container = ft.Container(
        content=ft.Column([
            ft.Text("Esperando análisis...", size=16, color=ft.Colors.BLUE_GREY_400),
        ], horizontal_alignment=ft.CrossAxisAlignment.START),
        padding=20,
        margin=ft.margin.only(top=20),
        width=500,
        border_radius=15,
        bgcolor=ft.Colors.BLUE_GREY_50
    )
    
    # --- FilePicker para simular la carga de la imagen ---
    
    def handle_file_pick(e: ft.FilePickerResultEvent):
        if e.files:
            # NOTA CLAVE: En la integración real, e.files[0].path se usaría para cargar la imagen con CV2.
            # Aquí, si es una ruta local, la usamos, sino, usamos un placeholder.
            img_path = e.files[0].path
            if img_path:
                img_display.src = img_path
            else:
                # Caso Web/Simulación (usamos una imagen de muestra)
                img_display.src = PLACEHOLDER_IMAGE_UPLOADED
            
            # Guardamos la ruta del archivo o la URL de simulación en el tooltip para usarla en run_analysis
            img_display.tooltip = img_display.src
            img_display.update()
        else:
            img_display.src = PLACEHOLDER_IMAGE_URL
            img_display.tooltip = "Rostro a analizar"
            img_display.update()

    file_picker = ft.FilePicker(on_result=handle_file_pick)
    page.overlay.append(file_picker)
    
    # --- Función principal de Diagnóstico (Coordina UI y Lógica) ---
    
    def run_analysis(e):
        # 1. Validación de la entrada
        try:
            chronological_age = int(txt_age.value)
            if chronological_age <= 0 or chronological_age > 120:
                raise ValueError("Edad inválida.")
        except ValueError:
            results_container.content = ft.Text("Por favor, ingresa una edad cronológica válida (1-120).", color=ft.Colors.RED_500)
            results_container.bgcolor = ft.Colors.RED_100
            page.update()
            return
        
        # Obtener la ruta de la imagen actual (de la carga o del placeholder)
        current_image_path = img_display.tooltip if img_display.tooltip != "Rostro a analizar" else PLACEHOLDER_IMAGE_UPLOADED
        
        # 2. Llamada a la Lógica de Predicción REAL
        estimated_age = predict_age_real(current_image_path, chronological_age)
        
        # 3. Cálculo del Índice de Envejecimiento
        aging_index = estimated_age - chronological_age
        
        # 4. Generación de Recomendaciones
        recommendations = get_recommendations(aging_index)

        # 5. Actualizar Resultados en la Interfaz
        
        # Determinación del color de acento
        if aging_index > 5:
            index_color = ft.Colors.RED_700
        elif aging_index >= 1:
            index_color = ft.Colors.ORANGE_700
        elif aging_index < -5:
            index_color = ft.Colors.GREEN_700
        else:
            index_color = ft.Colors.BLUE_700
            
        results_container.content = ft.Column([
            ft.Text("RESULTADOS DEL ANÁLISIS", size=20, weight="bold", color=ft.Colors.BLUE_GREY_900),
            ft.Divider(),
            ft.Row([
                ft.Text("Edad Cronológica:", weight="w500"),
                ft.Text(f"{chronological_age} años", weight="bold")
            ]),
            ft.Row([
                ft.Text("Edad Biológica Estimada:", weight="w500"),
                ft.Text(f"{estimated_age} años", size=18, weight="bold", color=index_color)
            ]),
            ft.Container(
                content=ft.Text(
                    f"Índice de Envejecimiento: {aging_index} años", 
                    size=22, 
                    weight="w700",
                    color=index_color
                ),
                padding=10,
                border_radius=8,
                bgcolor=index_color + "1A"
            ),
            ft.Divider(height=15),
            ft.Text("RECOMENDACIONES DE AUTOCUIDADO:", size=16, weight="bold", color=ft.Colors.GREY_800),
            ft.Column(recommendations, spacing=5),
        ], horizontal_alignment=ft.CrossAxisAlignment.START, spacing=10)
        
        results_container.bgcolor = ft.Colors.BLUE_GREY_50
        page.update()

    # --- Diseño de la Interfaz (Vista Principal) ---
    
    app_layout = ft.Container(
        width=550,
        padding=40,
        border_radius=20,
        shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.BLACK38),
        bgcolor=ft.Colors.WHITE,
        content=ft.Column(
            [
                ft.Text("Estimador de Envejecimiento Biológico", size=26, weight="bold", color=ft.Colors.CYAN_700),
                ft.Text("Análisis facial (Modelo Real/Simulado) basado en la edad y la imagen.", size=14, color=ft.Colors.BLUE_GREY_600),
                ft.Divider(height=10),
                
                # Sección de Carga de Imagen y Edad
                ft.Row([
                    # Columna Izquierda: Imagen
                    ft.Column([
                        img_display,
                        ft.ElevatedButton(
                            text="Cargar Foto del Rostro",
                            icon=ft.Icons.CAMERA_ALT_OUTLINED,
                            on_click=lambda _: file_picker.pick_files(
                                allow_multiple=False,
                                allowed_extensions=["jpg", "jpeg", "png"]
                            ),
                            style=ft.ButtonStyle(
                                bgcolor=ft.Colors.CYAN_500,
                                color=ft.Colors.WHITE,
                                shape=ft.RoundedRectangleBorder(radius=10),
                            )
                        )
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=15),
                    
                    # Columna Derecha: Entrada de Edad
                    ft.Column([
                        ft.Text("1. Ingresa tu edad:", size=16, weight="w500"),
                        txt_age,
                        ft.Text("2. Ejecutar el Modelo:", size=16, weight="w500"),
                        ft.ElevatedButton(
                            text="OBTENER DIAGNÓSTICO",
                            icon=ft.Icons.ANALYTICS_OUTLINED,
                            on_click=run_analysis,
                            height=50,
                            style=ft.ButtonStyle(
                                bgcolor=ft.Colors.LIGHT_BLUE_800,
                                color=ft.Colors.WHITE,
                                shape=ft.RoundedRectangleBorder(radius=10),
                                padding=ft.padding.only(left=20, right=20)
                            )
                        )
                    ], spacing=20)
                ], alignment=ft.MainAxisAlignment.SPACE_AROUND),
                
                ft.Divider(height=20),
                
                # Sección de Resultados y Recomendaciones
                results_container
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
    )
    
    page.add(app_layout)