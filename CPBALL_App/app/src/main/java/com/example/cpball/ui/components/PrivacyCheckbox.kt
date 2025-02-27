package com.example.cpball.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material.Checkbox
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.cpball.ui.theme.PrimaryOrange

@Composable
fun PrivacyCheckbox(checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
        Checkbox(
            checked = checked,
            onCheckedChange = onCheckedChange
        )
        Text(text = "I agree to the ")
        Text(
            text = "Privacy Policy",
            color = PrimaryOrange,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.clickable { /* Open Privacy Policy */ }
        )
    }
}
