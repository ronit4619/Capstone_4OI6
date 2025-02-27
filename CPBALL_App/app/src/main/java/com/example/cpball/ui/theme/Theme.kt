package com.example.cpball.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material.*
import androidx.compose.runtime.Composable

@Composable
fun CPBallTheme(content: @Composable () -> Unit) {
    val colors = lightColors(
        primary = PrimaryOrange,
        primaryVariant = BackgroundGradient,
        secondary = LightOrange,
        background = White,
        onPrimary = White,
        onBackground = TextGray
    )

    MaterialTheme(
        colors = colors,
        typography = Typography,
        shapes = Shapes,
        content = content
    )
}
