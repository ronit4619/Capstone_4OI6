package com.example.cpball.viewmodel
import com.example.cpball.model.LoginRequest

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.cpball.network.RetrofitClient  // Ensure this matches your actual package
import kotlinx.coroutines.launch

class AuthViewModel : ViewModel() {
    fun login(username: String, password: String, onResult: (Boolean, String?) -> Unit) {
        viewModelScope.launch {
            try {
                val response = RetrofitClient.api.login(LoginRequest(username, password))
                onResult(true, response.token) // Successful login
            } catch (e: Exception) {
                onResult(false, e.message) // Login failed
            }
        }
    }
}
