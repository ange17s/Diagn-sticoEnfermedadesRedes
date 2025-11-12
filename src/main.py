import flet as ft
from interfaz import main

if __name__ == "__main__":
    # Usa ft.app(target=main) para escritorio o ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=8000) para web
    ft.app(target=main)