Android Framework
==================

## [ContentProviders]
--------------------

- Interface to connect structured data in one process to code in another process.
- Wraps IPC and data transfer.
- Client uses ```ContentResolver``` instance to communicate with 
```ContentProvider``` instance in the Service.
- provider object :
	* presents data to external apps as 1/more tables
   	* receives data request from clients
   	* performs the requested action
   	* returns the result

### Accessing a provider (Cursor / CursorLoader)

- Data accessed through ```ContentResolver```
- Provides basic *CRUD* (create / retrieve / update / delete) functionality
- **URI** used to resolve the provider by comparing the **Authority** from a system table 
of known providers


#### ```Cursor```-based data access
- Runs on UI Thread
- Query performed using : ```query(Uri,projection,selection,selectionArgs,sortOrder)```,
where:
	* Uri - ```FROM``` table name : (ProviderName + table path : ```content://user_dictionary/words```
	* projection - column names
	* selection - ```WHERE``` clause
	* sortOrder - ```ORDER BY``` clause

```java   
// Queries the user dictionary and returns results
Cursor mCursor = getContentResolver().query(
    UserDictionary.Words.CONTENT_URI,   // The content URI of the words table
    mProjection,                        // The columns to return for each row
    mSelectionClause                    // Selection criteria
    mSelectionArgs,                     // Selection criteria
    mSortOrder);                        // The sort order for the returned rows
if (null == mCursor) {
	..
}
else if( mCursor.getCount() < 1){
	..
}
else{
	//logic here
}
```

#### ```CursorLoader```-based data access

- used to run an *async* query against a ```ContentProvider```, and returns the
result to an ```Activity / FragmentActivity ```.
- Steps:
	* **IMPLEMENT** : ```LoaderManager.LoaderCallbacks<Cursor>```
	* **INIT** : ```getSupportLoaderManager().initLoader()``` in ```onCreateView()```
	* **QUERY** : override ```onCreateLoader``` and return ```CursorLoader```
	* **RESULTS** : override ```onLoadFinished``` and do : ```Adapter.changeCursor(mCursor)```

### [android.provider]

- Framework-level support for some ```ContentProviders```
- ```android.providers``` provides convenience classes to access these
	* Calendar
	* Contacts
	* Documents
	* MediaStore (Images / Video / Files)
	* SearchRecentSuggestions
	* SyncState
	* Telephony (SMS / MMS)
	* UserDictionary
	* Voicemail



1. [ Android - ContentProviders](http://developer.android.com/guide/topics/providers/content-providers.html)   
2. [ Android - list of ContentProviders](http://developer.android.com/reference/android/provider/package-summary.html)
