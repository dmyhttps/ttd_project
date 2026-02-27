import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class ApiService {
  // Change this to your backend URL
  static const String baseUrl = 'http://localhost:8000';
  // For Production: 'https://your-deployed-backend.com'
  
  static const String _tokenKey = 'auth_token';
  static const String _userIdKey = 'user_id';

  /// Get stored JWT token
  static Future<String?> getToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_tokenKey);
  }

  /// Save JWT token after login
  static Future<void> saveToken(String token, String userId) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenKey, token);
    await prefs.setString(_userIdKey, userId);
  }

  /// Clear token on logout
  static Future<void> clearToken() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_userIdKey);
  }

  /// Get stored user ID
  static Future<String?> getUserId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_userIdKey);
  }

  // =====================================================================
  // AUTHENTICATION ENDPOINTS
  // =====================================================================

  /// Sign up new user
  static Future<Map<String, dynamic>> signUp(
    String email,
    String password,
    String fullName,
  ) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/api/auth/signup'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'email': email,
          'password': password,
          'full_name': fullName,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        await saveToken(data['access_token'], data['user_id']);
        return {'success': true, 'user_id': data['user_id']};
      } else {
        final error = jsonDecode(response.body);
        return {'success': false, 'error': error['detail'] ?? 'Sign up failed'};
      }
    } catch (e) {
      return {'success': false, 'error': 'Network error: $e'};
    }
  }

  /// Sign in existing user
  static Future<Map<String, dynamic>> signIn(
    String email,
    String password,
  ) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/api/auth/signin'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'email': email,
          'password': password,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        await saveToken(data['access_token'], data['user_id']);
        return {'success': true, 'user_id': data['user_id']};
      } else {
        final error = jsonDecode(response.body);
        return {'success': false, 'error': error['detail'] ?? 'Sign in failed'};
      }
    } catch (e) {
      return {'success': false, 'error': 'Network error: $e'};
    }
  }

  // =====================================================================
  // USER ENDPOINTS
  // =====================================================================

  /// Get user profile
  static Future<Map<String, dynamic>> getUserProfile() async {
    try {
      final token = await getToken();
      if (token == null) {
        return {'success': false, 'error': 'Not authenticated'};
      }

      final response = await http.get(
        Uri.parse('$baseUrl/api/user/profile?token=$token'),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return {'success': true, 'user': data};
      } else {
        return {'success': false, 'error': 'Failed to get profile'};
      }
    } catch (e) {
      return {'success': false, 'error': 'Network error: $e'};
    }
  }

  // =====================================================================
  // PREDICTION ENDPOINTS
  // =====================================================================

  /// Make a threat prediction
  static Future<Map<String, dynamic>> makePrediction(
    String inputText,
    String inputType, // 'text', 'pdf', 'txt', 'audio'
  ) async {
    try {
      final token = await getToken();
      if (token == null) {
        return {'success': false, 'error': 'Not authenticated'};
      }

      final response = await http.post(
        Uri.parse('$baseUrl/api/predict?token=$token'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'input_text': inputText,
          'input_type': inputType,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return {'success': true, 'data': data};
      } else {
        final error = jsonDecode(response.body);
        return {'success': false, 'error': error['detail'] ?? 'Prediction failed'};
      }
    } catch (e) {
      return {'success': false, 'error': 'Network error: $e'};
    }
  }

  // =====================================================================
  // HISTORY & FILTERING ENDPOINTS
  // =====================================================================

  /// Get user's prediction history with optional filters
  static Future<Map<String, dynamic>> getHistory({
    String? inputType,
    String? prediction,
    int limit = 50,
  }) async {
    try {
      final token = await getToken();
      if (token == null) {
        return {'success': false, 'error': 'Not authenticated'};
      }

      String url = '$baseUrl/api/history?token=$token&limit=$limit';
      
      if (inputType != null) {
        url += '&input_type=$inputType';
      }
      if (prediction != null) {
        url += '&prediction=$prediction';
      }

      final response = await http.get(Uri.parse(url));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return {
          'success': true,
          'logs': data['logs'] ?? [],
          'count': data['count'] ?? 0
        };
      } else {
        return {'success': false, 'error': 'Failed to get history'};
      }
    } catch (e) {
      return {'success': false, 'error': 'Network error: $e'};
    }
  }

  /// Get statistics by input type
  static Future<Map<String, dynamic>> getStatistics() async {
    try {
      final token = await getToken();
      if (token == null) {
        return {'success': false, 'error': 'Not authenticated'};
      }

      final response = await http.get(
        Uri.parse('$baseUrl/api/statistics?token=$token'),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return {'success': true, 'statistics': data['statistics']};
      } else {
        return {'success': false, 'error': 'Failed to get statistics'};
      }
    } catch (e) {
      return {'success': false, 'error': 'Network error: $e'};
    }
  }

  /// Delete a specific log entry
  static Future<Map<String, dynamic>> deleteLog(int logId) async {
    try {
      final token = await getToken();
      if (token == null) {
        return {'success': false, 'error': 'Not authenticated'};
      }

      final response = await http.delete(
        Uri.parse('$baseUrl/api/history/$logId?token=$token'),
      );

      if (response.statusCode == 200) {
        return {'success': true};
      } else {
        return {'success': false, 'error': 'Failed to delete log'};
      }
    } catch (e) {
      return {'success': false, 'error': 'Network error: $e'};
    }
  }
}
