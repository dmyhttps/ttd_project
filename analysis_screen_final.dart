import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:final_year_prohect/screens/auth_screen.dart';
import 'package:final_year_prohect/services/api_service.dart';

class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  final TextEditingController _textController = TextEditingController();
  
  String? _result;
  double? _threatScore;
  bool _isAnalyzing = false;
  String? _detectionMethod;
  List<String>? _threatIndicators;

  // History and filtering
  List<dynamic> _history = [];
  String? _selectedFilter;
  bool _loadingHistory = false;
  Map<String, dynamic>? _userProfile;

  @override
  void initState() {
    super.initState();
    _loadUserProfile();
    _loadHistory();
  }

  Future<void> _loadUserProfile() async {
    final result = await ApiService.getUserProfile();
    if (result['success']) {
      setState(() {
        _userProfile = result['user'];
      });
    }
  }

  Future<void> _loadHistory({String? inputType}) async {
    setState(() {
      _loadingHistory = true;
    });

    final result = await ApiService.getHistory(inputType: inputType);
    if (result['success']) {
      setState(() {
        _history = result['logs'] ?? [];
        _loadingHistory = false;
      });
    } else {
      setState(() {
        _loadingHistory = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(result['error'] ?? 'Failed to load history'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _analyzeContent() async {
    if (_textController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter some text to analyze')),
      );
      return;
    }

    setState(() {
      _isAnalyzing = true;
      _result = null;
    });

    final result = await ApiService.makePrediction(
      _textController.text,
      'text',
    );

    setState(() {
      _isAnalyzing = false;
    });

    if (result['success']) {
      final data = result['data'];
      setState(() {
        _result = data['prediction'] == 'threatening'
            ? 'Threat Detected'
            : 'Safe';
        _threatScore = data['confidence'] / 100;
        _detectionMethod = data['detection_method'];
        _threatIndicators = List<String>.from(data['threat_indicators'] ?? []);
      });

      _loadHistory(inputType: _selectedFilter);
      _textController.clear();
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(result['error'] ?? 'Analysis failed'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _deleteLog(int logId) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Log'),
        content: const Text('Are you sure you want to delete this log?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      final result = await ApiService.deleteLog(logId);
      if (result['success']) {
        _loadHistory(inputType: _selectedFilter);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Log deleted successfully')),
          );
        }
      }
    }
  }

  void _logout() async {
    await ApiService.clearToken();
    if (mounted) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => const AuthScreen()),
      );
    }
  }

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Threat Detection System'),
          elevation: 2,
          actions: [
            if (_userProfile != null)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Center(
                  child: Text(
                    _userProfile!['email'] ?? 'User',
                    style: const TextStyle(fontSize: 14),
                  ),
                ),
              ),
            IconButton(
              icon: const Icon(Icons.logout),
              tooltip: 'Logout',
              onPressed: _logout,
            ),
          ],
          bottom: const TabBar(
            tabs: [
              Tab(text: 'Analyze'),
              Tab(text: 'History'),
            ],
          ),
        ),
        body: TabBarView(
          children: [
            // ANALYZE TAB
            Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24.0),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 600),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Text(
                        'Analyze Content for Threats',
                        style: Theme.of(context).textTheme.headlineSmall,
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 32),
                      TextField(
                        controller: _textController,
                        maxLines: 8,
                        enabled: !_isAnalyzing,
                        decoration: InputDecoration(
                          hintText: 'Enter text to analyze...',
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                          contentPadding: const EdgeInsets.all(16),
                        ),
                      ),
                      const SizedBox(height: 24),
                      FilledButton(
                        onPressed: _isAnalyzing ? null : _analyzeContent,
                        style: FilledButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                        ),
                        child: _isAnalyzing
                            ? const SizedBox(
                                height: 20,
                                width: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  valueColor:
                                      AlwaysStoppedAnimation(Colors.white),
                                ),
                              )
                            : const Text('Analyze'),
                      ),
                      const SizedBox(height: 32),
                      // Results
                      if (_result != null) ...[
                        Card(
                          elevation: 4,
                          child: Padding(
                            padding: const EdgeInsets.all(16.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Icon(
                                      _result == 'Threat Detected'
                                          ? Icons.warning
                                          : Icons.check_circle,
                                      color: _result == 'Threat Detected'
                                          ? Colors.red
                                          : Colors.green,
                                      size: 32,
                                    ),
                                    const SizedBox(width: 16),
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            _result!,
                                            style: Theme.of(context)
                                                .textTheme
                                                .titleLarge
                                                ?.copyWith(
                                                  fontWeight: FontWeight.bold,
                                                  color: _result ==
                                                          'Threat Detected'
                                                      ? Colors.red
                                                      : Colors.green,
                                                ),
                                          ),
                                          const SizedBox(height: 8),
                                          Text(
                                            'Confidence: ${(_threatScore! * 100).toStringAsFixed(2)}%',
                                            style: Theme.of(context)
                                                .textTheme
                                                .bodyMedium,
                                          ),
                                        ],
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 16),
                                Divider(
                                  color: Theme.of(context)
                                      .colorScheme
                                      .outline,
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  'Detection Method: $_detectionMethod',
                                  style: Theme.of(context).textTheme.bodySmall,
                                ),
                                if (_threatIndicators != null &&
                                    _threatIndicators!.isNotEmpty) ...[
                                  const SizedBox(height: 8),
                                  Text(
                                    'Threat Indicators: ${_threatIndicators!.join(', ')}',
                                    style: Theme.of(context).textTheme.bodySmall,
                                  ),
                                ],
                              ],
                            ),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
            // HISTORY TAB
            Column(
              children: [
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    children: [
                      const Text('Filter by type:'),
                      const SizedBox(width: 12),
                      Expanded(
                        child: SingleChildScrollView(
                          scrollDirection: Axis.horizontal,
                          child: Row(
                            children: ['All', 'text', 'pdf', 'txt', 'audio']
                                .map((type) {
                              final isSelected = _selectedFilter == null && type == 'All' ||
                                  _selectedFilter == type;
                              return Padding(
                                padding: const EdgeInsets.only(right: 8.0),
                                child: FilterChip(
                                  label: Text(type),
                                  selected: isSelected,
                                  onSelected: (selected) {
                                    setState(() {
                                      if (type == 'All') {
                                        _selectedFilter = null;
                                      } else {
                                        _selectedFilter =
                                            selected ? type : null;
                                      }
                                    });
                                    _loadHistory(inputType: _selectedFilter);
                                  },
                                ),
                              );
                            }).toList(),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: _loadingHistory
                      ? const Center(
                          child: CircularProgressIndicator(),
                        )
                      : _history.isEmpty
                          ? Center(
                              child: Text(
                                'No history found',
                                style: Theme.of(context).textTheme.bodyLarge,
                              ),
                            )
                          : ListView.builder(
                              itemCount: _history.length,
                              itemBuilder: (context, index) {
                                final log = _history[index];
                                final dateTime = DateTime.parse(log['created_at']);
                                final formattedDate =
                                    DateFormat('MMM dd, yyyy HH:mm')
                                        .format(dateTime);

                                return Card(
                                  margin: const EdgeInsets.symmetric(
                                    horizontal: 16,
                                    vertical: 8,
                                  ),
                                  child: ListTile(
                                    title: Text(
                                      '${log['prediction'].toString().toUpperCase()} - ${log['confidence']?.toStringAsFixed(2)}%',
                                    ),
                                    subtitle: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        const SizedBox(height: 8),
                                        Text(
                                          'Type: ${log['input_type']} | Method: ${log['detection_method']}',
                                          style: Theme.of(context)
                                              .textTheme
                                              .bodySmall,
                                        ),
                                        Text(
                                          formattedDate,
                                          style: Theme.of(context)
                                              .textTheme
                                              .bodySmall,
                                        ),
                                      ],
                                    ),
                                    trailing: IconButton(
                                      icon: const Icon(Icons.delete,
                                          color: Colors.red),
                                      onPressed: () =>
                                          _deleteLog(log['id']),
                                    ),
                                  ),
                                );
                              },
                            ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
