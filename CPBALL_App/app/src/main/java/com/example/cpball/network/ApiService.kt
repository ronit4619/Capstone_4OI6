package com.example.cpball.network
import com.example.cpball.model.LoginRequest

import com.example.cpball.model.LoginResponse
import retrofit2.http.Body
import retrofit2.http.POST

interface ApiService {
    @POST("login")
    suspend fun login(@Body request: LoginRequest): LoginResponse
}
