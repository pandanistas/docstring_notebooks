error_descriptions = {
    "GL01": {
        "description": "Docstring text (summary) should start in the line immediately after the opening quotes (not in the same line, or leaving a blank line in between.)",
        "references": "- [Contributing Docstring - Section 2: Extended Summary](https://pandas.io/docs/development/contributing_docstring.html#section-2-extended-summary)",
        "bad_example": '''```python
        
        def assert_categorical_equal(
            left, right, check_dtype=True, check_category_order=True, obj="Categorical"
        ):
            """Test that Categoricals are equivalent.

            Parameters
            ----------
            left : Categorical
            ...
            """
            pass
        ```
        ''',
        "good_example": '''```python
        
        def assert_categorical_equal(
            left, right, check_dtype=True, check_category_order=True, obj="Categorical"
        ):
            """
            Test that Categoricals are equivalent.

            Parameters
            ----------
            left : Categorical
            ...
            """
            pass
        
        ```
        '''
    },
    "GL02": {
        "description": "Closing quotes should be placed in the line after the last text in the docstring (do not close the quotes in the same line as the text, or leave a blank line between the last text and the quotes)",
        "references": "",
        "bad_example": '''```python
        
        def unstack():
            """
            Pivot a row index to columns.

            When using a MultiIndex, a level can be pivoted so each value in
            the index becomes a column. This is especially useful when a subindex
            is repeated for the main index, and data is easier to visualize as a
            pivot table.

            The index level will be automatically removed from the index when added
            as columns."""
            pass
        
        ```
        ''',
        "good_example": '''```python
        
        def unstack():
            """
            Pivot a row index to columns.

            When using a MultiIndex, a level can be pivoted so each value in
            the index becomes a column. This is especially useful when a subindex
            is repeated for the main index, and data is easier to visualize as a
            pivot table.

            The index level will be automatically removed from the index when added
            as columns.
            """
            pass
        
        ```
        '''
    },
    "GL08": {
        "description": "The object does not have a docstring",
        "references": "- [pandas docstring guide](https://pandas.io/docs/development/contributing_docstring.html)",
        "bad_example": '''```python
        
        @property
        def right(self):
            return self._data._right
        
        ```
        ''',
        "good_example": '''```python
        
        @property
        def right(self):
            """
            Return the right endpoints of each Interval in the IntervalIndex as
            an Index.
            """
            return self._data._right
        ```
        '''
    },
    "SS06": {
        "description": "Summary should fit in a single line",
        "references": "",
        "bad_example": '''```python
        
        def duplicated(self, subset=None, keep="first"):
            """
            Return boolean Series denoting duplicate rows, optionally only
            considering certain columns.
            """
            ...
        
        ```
        ''',
        "good_example": '''```python
        
        def duplicated(self, subset=None, keep="first"):
            """
            Return boolean Series denoting duplicate rows.

            Considering certain columns is optional.
            """
            ...
        
        ```
        '''
    },
    "ES01": {
        "description": "No extended summary found",
        "references": "- [Extended summary](https://pandas.io/docs/development/contributing_docstring.html#section-2-extended-summary)",
        "bad_example": '''```python
        
        def unstack():
            """
            Pivot a row index to columns.
            """
            pass
        
        
        ```
        ''',
        "good_example": '''```python
        
        def unstack():
            """
            Pivot a row index to columns.

            When using a MultiIndex, a level can be pivoted so each value in
            the index becomes a column. This is especially useful when a subindex
            is repeated for the main index, and data is easier to visualize as a
            pivot table.

            The index level will be automatically removed from the index when added
            as columns.
            """
            pass
        
        
        ```
        '''
    },
    "PR01": {
        "description": "Parameters {missing_params} not documented",
        "references": "",
        "bad_example": '''```python
        
        class Series:
            def plot(self, kind, **kwargs):
                """
                Generate a plot.

                Render the data in the Series as a matplotlib plot of the
                specified kind.

                Note the blank line between the parameters title and the first
                parameter. Also, note that after the name of the parameter `kind`
                and before the colon, a space is missing.

                Also, note that the parameter descriptions do not start with a
                capital letter, and do not finish with a dot.

                Finally, the `**kwargs` parameter is missing.

                Parameters
                ----------

                kind: str
                    kind of matplotlib plot
                """
                pass
        
        
        ```
        ''',
        "good_example": '''```python
        
        # We need to add **kwargs** to the docstring
        class Series:
            def plot(self, kind, color='blue', **kwargs):
                """
                Generate a plot.

                Render the data in the Series as a matplotlib plot of the
                specified kind.

                Parameters
                ----------
                kind : str
                    Kind of matplotlib plot.
                color : str, default 'blue'
                    Color name or rgb code.
                **kwargs
                    These parameters will be passed to the matplotlib plotting
                    function.
                """
                pass
        
        ```
        '''
    },
    "PR02": {
        "description": "",
        "references": "",
        "bad_example": '''```python
        
        # kwargs is not recognized as a parameter. It should be **kwargs.
        def astype(self, dtype, copy=True, errors="raise", **kwargs):
            """
                Cast a pandas object to a specified dtype ``dtype``.
                Parameters
                ----------
                dtype : data type, or dict of column name -> data type
                    Use a numpy.dtype or Python type to cast entire pandas object to
                    the same type. Alternatively, use {col: dtype, ...}, where col is a
                    column label and dtype is a numpy.dtype or Python type to cast one
                    or more of the DataFrame's columns to column-specific types.
                copy : bool, default True
                    Return a copy when ``copy=True`` (be very careful setting
                    ``copy=False`` as changes to values then may propagate to other
                    pandas objects).
                errors : {'raise', 'ignore'}, default 'raise'
                    Control raising of exceptions on invalid data for provided dtype.
                    - ``raise`` : allow exceptions to be raised
                    - ``ignore`` : suppress exceptions. On error return original object.
                    .. versionadded:: 0.20.0
                kwargs : keyword arguments to pass on to the constructor
                Returns
                -------
                casted : same type as caller
            """
            ...
        
        
        ```
        ''',
        "good_example": '''```python
        
        # Change kwargs to **kwargs
        def astype(self, dtype, copy=True, errors="raise", **kwargs):
            """
                Cast a pandas object to a specified dtype ``dtype``.
                Parameters
                ----------
                dtype : data type, or dict of column name -> data type
                    Use a numpy.dtype or Python type to cast entire pandas object to
                    the same type. Alternatively, use {col: dtype, ...}, where col is a
                    column label and dtype is a numpy.dtype or Python type to cast one
                    or more of the DataFrame's columns to column-specific types.
                copy : bool, default True
                    Return a copy when ``copy=True`` (be very careful setting
                    ``copy=False`` as changes to values then may propagate to other
                    pandas objects).
                errors : {'raise', 'ignore'}, default 'raise'
                    Control raising of exceptions on invalid data for provided dtype.
                    - ``raise`` : allow exceptions to be raised
                    - ``ignore`` : suppress exceptions. On error return original object.
                    .. versionadded:: 0.20.0
                **kwargs : keyword arguments to pass on to the constructor
                Returns
                -------
                casted : same type as caller
            """
            ...
        
        ```
        '''
    },
    
    "PR06": {
        "description": 'Parameter "{param_name}" type should use "{right_type}" instead of "{wrong_type}"',
        "references": "",
        "bad_example": '''```python
        
        # The code below would output an error "Parameter 'path' type should use 'str' instead of 'string'.
        def read_spss(
            path: Union[str, Path],
            usecols: Optional[Sequence[str]] = None,
            convert_categoricals: bool = True,
        ) -> DataFrame:
            """
            Load an SPSS file from the file path, returning a DataFrame.

            .. versionadded:: 0.25.0

            Parameters
            ----------
            path : string or Path
                File path.
            usecols : list-like, optional
                Return a subset of the columns. If None, return all columns.
            convert_categoricals : bool, default is True
                Convert categorical columns into pd.Categorical.

            Returns
            -------
            DataFrame
            """
        
        
        ```
        ''',
        "good_example": '''```python
        
        def read_spss(
            path: Union[str, Path],
            usecols: Optional[Sequence[str]] = None,
            convert_categoricals: bool = True,
        ) -> DataFrame:
            """
            Load an SPSS file from the file path, returning a DataFrame.

            .. versionadded:: 0.25.0

            Parameters
            ----------
            path : str or Path
                File path.
            usecols : list-like, optional
                Return a subset of the columns. If None, return all columns.
            convert_categoricals : bool, default is True
                Convert categorical columns into pd.Categorical.

            Returns
            -------
            DataFrame
            """
        
        ```
        '''
    },
    "PR07": {
        "description": 'Parameter "{param_name}" has no description',
        "references": "",
        "bad_example": '''```python
        
        # In the example below, the parameter axis is missing a description:
        def _get_counts_nanvar(
            value_counts: Tuple[int],
            mask: Optional[np.ndarray],
            axis: Optional[int],
            ddof: int,
            dtype=float,
        ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
            """ Get the count of non-null values along an axis, accounting
            for degrees of freedom.

            Parameters
            ----------
            values_shape : Tuple[int]
                shape tuple from values ndarray, used if mask is None
            mask : Optional[ndarray[bool]]
                locations in values that should be considered missing
            axis : Optional[int]
            ddof : int
                degrees of freedom
            dtype : type, optional
                type to use for count

            Returns
            -------
            count : scalar or array
            d : scalar or array
            """
        
        ```
        ''',
        "good_example": '''```python
        
        def _get_counts_nanvar(
            value_counts: Tuple[int],
            mask: Optional[np.ndarray],
            axis: Optional[int],
            ddof: int,
            dtype=float,
        ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
            """ Get the count of non-null values along an axis, accounting
            for degrees of freedom.

            Parameters
            ----------
            values_shape : Tuple[int]
                shape tuple from values ndarray, used if mask is None
            mask : Optional[ndarray[bool]]
                locations in values that should be considered missing
            axis : Optional[int]
                axis to count along
            ddof : int
                degrees of freedom
            dtype : type, optional
                type to use for count

            Returns
            -------
            count : scalar or array
            d : scalar or array
            """
        
        
        ```
        '''
    },
    
    "PR08": {
        "description": 'Parameter "{param_name}" description should start with a capital letter',
        "references": "",
        "bad_example": '''```python
        
        # The description of the parameter axis does not start with a capital letter:
        def take_nd(
            arr, indexer, axis: int = 0, out=None, fill_value=np.nan, allow_fill: bool = True
        ):
            """
            Specialized Cython take which sets NaN values in one pass

            This dispatches to ``take`` defined on ExtensionArrays. It does not
            currently dispatch to ``SparseArray.take`` for sparse ``arr``.

            Parameters
            ----------
            arr : array-like
                Input array.
            indexer : ndarray
                1-D array of indices to take, subarrays corresponding to -1 value
                indices are filed with fill_value
            axis : int, default 0
                axis to take from
            out : ndarray or None, default None
                Optional output array, must be appropriate type to hold input and
                fill_value together, if indexer has any -1 value entries; call
                maybe_promote to determine this type for any fill_value
            fill_value : any, default np.nan
                Fill value to replace -1 values with
            allow_fill : boolean, default True
                If False, indexer is assumed to contain no -1 values so no filling
                will be done.  This short-circuits computation of a mask.  Result is
                undefined if allow_fill == False and -1 is present in indexer.

            Returns
            -------
            subarray : array-like
                May be the same type as the input, or cast to an ndarray.
            """
        
        
        ```
        ''',
        "good_example": '''```python
        
        def take_nd(
            arr, indexer, axis: int = 0, out=None, fill_value=np.nan, allow_fill: bool = True
        ):
            """
            Specialized Cython take which sets NaN values in one pass

            This dispatches to ``take`` defined on ExtensionArrays. It does not
            currently dispatch to ``SparseArray.take`` for sparse ``arr``.

            Parameters
            ----------
            arr : array-like
                Input array.
            indexer : ndarray
                1-D array of indices to take, subarrays corresponding to -1 value
                indices are filed with fill_value
            axis : int, default 0
                Axis to take from
            out : ndarray or None, default None
                Optional output array, must be appropriate type to hold input and
                fill_value together, if indexer has any -1 value entries; call
                maybe_promote to determine this type for any fill_value
            fill_value : any, default np.nan
                Fill value to replace -1 values with
            allow_fill : boolean, default True
                If False, indexer is assumed to contain no -1 values so no filling
                will be done.  This short-circuits computation of a mask.  Result is
                undefined if allow_fill == False and -1 is present in indexer.

            Returns
            -------
            subarray : array-like
                May be the same type as the input, or cast to an ndarray.
            """
        
        ```
        '''
    },
    "PR09": {
        "description": 'Parameter description should finish with "."',
        "references": "",
        "bad_example": '''```python
        
        def cumsum(self, axis=0, *args, **kwargs):
            # The description of the parameter axis does not finish with ".":
            """
            Cumulative sum of non-NA/null values.

            When performing the cumulative summation, any non-NA/null values will
            be skipped. The resulting SparseArray will preserve the locations of
            NaN values, but the fill value will be `np.nan` regardless.

            Parameters
            ----------
            axis : int or None
                Axis over which to perform the cumulative summation. If None,
                perform cumulative summation over flattened array

            Returns
            -------
            cumsum : SparseArray
            """
        
        ```
        ''',
        "good_example": '''```python
        
        def cumsum(self, axis=0, *args, **kwargs):
            """
            Cumulative sum of non-NA/null values.

            When performing the cumulative summation, any non-NA/null values will
            be skipped. The resulting SparseArray will preserve the locations of
            NaN values, but the fill value will be `np.nan` regardless.

            Parameters
            ----------
            axis : int or None
                Axis over which to perform the cumulative summation. If None,
                perform cumulative summation over flattened array.

            Returns
            -------
            cumsum : SparseArray
            """
        
        ```
        '''
    },
    "RT02": {
        "description": "The first line of the Returns section should contain only the type, unless multiple values are being returned",
        "references": "",
        "bad_example": '''```python
        
        # The first line of the Returns section should contain only the type:
        def is_overlapping(self):
            """
            Return True if the IntervalIndex has overlapping intervals, else False.

            Two intervals overlap if they share a common point, including closed
            endpoints. Intervals that only have an open endpoint in common do not
            overlap.

            .. versionadded:: 0.24.0

            Returns
            -------
            bool : Boolean indicating if the IntervalIndex has overlapping intervals.
        
        
        ```
        ''',
        "good_example": '''```python
        
        def is_overlapping(self):
            """
            Return True if the IntervalIndex has overlapping intervals, else False.

            Two intervals overlap if they share a common point, including closed
            endpoints. Intervals that only have an open endpoint in common do not
            overlap.

            .. versionadded:: 0.24.0

            Returns
            -------
            bool
                Boolean indicating if the IntervalIndex has overlapping intervals.
        
        ```
        '''
    },
     "RT03": {
        "description": "Return value has no description",
        "references": "",
        "bad_example": '''```python
        
        def is_overlapping(self):
            """
            Return True if the IntervalIndex has overlapping intervals, else False.

            Two intervals overlap if they share a common point, including closed
            endpoints. Intervals that only have an open endpoint in common do not
            overlap.

            .. versionadded:: 0.24.0

            Returns
            -------
            bool
        
        ```
        ''',
        "good_example": '''```python
        
        def is_overlapping(self):
            """
            Return True if the IntervalIndex has overlapping intervals, else False.

            Two intervals overlap if they share a common point, including closed
            endpoints. Intervals that only have an open endpoint in common do not
            overlap.

            .. versionadded:: 0.24.0

            Returns
            -------
            bool
                Boolean indicating if the IntervalIndex has overlapping intervals.
        
        ```
        '''
    },
    "YD01": {
        "description": "No Yields section found",
        "references": "",
        "bad_example": '''```python
        
        def __iter__(self):
            """
            Return an iterator over the boxed values
            """
            ...
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                converted = tslib.ints_to_pydatetime(
                    data[start_i:end_i], tz=self.tz, freq=self.freq, box="timestamp"
                )
                for v in converted:
                    yield v
        
        ```
        ''',
        "good_example": '''```python
        
        def __iter__(self):
            """
            Return an iterator over the boxed values

            Yields
            ------
            tstamp : Timestamp
            """
            ...
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                converted = tslib.ints_to_pydatetime(
                    data[start_i:end_i], tz=self.tz, freq=self.freq, box="timestamp"
                )
                for v in converted:
                    yield v
        
        ```
        '''
    },
    "SA04": {
        "description": "Missing description for see also",
        "references": "",
        "bad_example": '''```python
        
        def mean(self, skipna=True):
            """
            Return the mean value of the Array.

            .. versionadded:: 0.25.0

            Parameters
            ----------
            skipna : bool, default True
                Whether to ignore any NaT elements.

            Returns
            -------
            scalar
                Timestamp or Timedelta.

            See Also
            --------
            numpy.ndarray.mean
            Series.mean

            Notes
            -----
            mean is only defined for Datetime and Timedelta dtypes, not for Period.
            """
        
        ```
        ''',
        "good_example": '''```python
        
        def mean(self, skipna=True):
            """
            Return the mean value of the Array.

            .. versionadded:: 0.25.0

            Parameters
            ----------
            skipna : bool, default True
                Whether to ignore any NaT elements.

            Returns
            -------
            scalar
                Timestamp or Timedelta.

            See Also
            --------
            numpy.ndarray.mean : Returns the average of array elements along a given axis.
            Series.mean : Return the mean value in a Series.

            Notes
            -----
            mean is only defined for Datetime and Timedelta dtypes, not for Period.
            """
        
        ```
        '''
    },
    
    "EX02": {
        "description": '\n\n'.join([
            "Examples do not pass tests: {doctest_log}. \n\nTo see exactly which test fails, you can run:",
            """```bash
            
            python scripts/validate_docstrings.py pandas.Series.str.split```""",
            "For example you'll see that one of the failed tests is:",
            """```python
            ################################################################################
            ################################### Doctests ###################################
            ################################################################################

            **********************************************************************
            Line 50, in pandas.Series.str.split
            Failed example:
                s = pd.Series(["this is a regular sentence",
                               "https://docs.python.org/3/tutorial/index.html",
                               np.nan])
            Expected:
                0                       this is a regular sentence
                1    https://docs.python.org/3/tutorial/index.html
                2                                              NaN
                dtype: object
            Got nothing```""",
            "When you check the docs, it looks like this:",
            """```
               Examples
                --------
                >>> s = pd.Series(["this is a regular sentence",
                ...                "https://docs.python.org/3/tutorial/index.html",
                ...                np.nan])
                0                       this is a regular sentence
                1    https://docs.python.org/3/tutorial/index.html
                2                                              NaN
                dtype: object```
            """,
            "This docs does not pass the test because, the way it's written, it seems that it is expecting",
            """
                0                       this is a regular sentence
                1    https://docs.python.org/3/tutorial/index.html
                2                                              NaN
                dtype: object
            """,
            "When we run",
            """```python
                s = pd.Series(["this is a regular sentence",
                    "https://docs.python.org/3/tutorial/index.html",
                    np.nan])```
            """,
            "Which is false. For the code to give us the expected output, we need to run s separately, by doing",
            """```python
            >>> s
            ```
            """,
            "So the new doc should look like this:",
            """```python
             Examples
            --------
            >>> s = pd.Series(["this is a regular sentence",
            ...                "https://docs.python.org/3/tutorial/index.html",
            ...                np.nan])

            >>> s
            0                       this is a regular sentence
            1    https://docs.python.org/3/tutorial/index.html
            2                                              NaN
            dtype: object
            ```
            """
        ]),
        "references": "",
        "bad_example": '''```python
        
        
        ```
        ''',
        "good_example": '''```python
        
        
        ```
        '''
    },
    "EX03": {
        "description": "flake8 error: {error_code} {error_message}{times_happening}",
        "references": "- [Working on Flake8 Errors](https://github.com/pandanistas/docstring_notebooks/wiki/flake8-Errors) ",
        "bad_example": '''```python
        
        
        ```
        ''',
        "good_example": '''```python
        
        
        ```
        '''
    },
    "SA01": {
        "description": "See Also section not found",
        "references": "",
        "bad_example": '''```python
        
        def argsort(
            self, ascending: bool = True, kind: str = "quicksort", *args, **kwargs
        ) -> np.ndarray:
            """
            Return the indices that would sort this array.

            Parameters
            ----------
            ascending : bool, default True
                Whether the indices should result in an ascending
                or descending sort.
            kind : {'quicksort', 'mergesort', 'heapsort'}, optional
                Sorting algorithm.
            *args, **kwargs:
                passed through to :func:`numpy.argsort`.

            Returns
            -------
            ndarray
                Array of indices that sort ``self``. If NaN values are contained,
                NaN values are placed at the end.
            """
        
        
        ```
        ''',
        "good_example": '''```python
        
        def argsort(
            self, ascending: bool = True, kind: str = "quicksort", *args, **kwargs
        ) -> np.ndarray:
            """
            Return the indices that would sort this array.

            Parameters
            ----------
            ascending : bool, default True
                Whether the indices should result in an ascending
                or descending sort.
            kind : {'quicksort', 'mergesort', 'heapsort'}, optional
                Sorting algorithm.
            *args, **kwargs:
                passed through to :func:`numpy.argsort`.

            Returns
            -------
            ndarray
                Array of indices that sort ``self``. If NaN values are contained,
                NaN values are placed at the end.

            See Also
            --------
            numpy.argsort : Sorting implementation used internally.
            """
        
        
        ```
        '''
    },
    "SA04": {
        "description": "Missing description for See Also {reference_name} reference",
        "references": "",
        "bad_example": '''```python
        
        def isna(self):
            """
            Detect missing values

            Missing values (-1 in .codes) are detected.

            Returns
            -------
            a boolean array of whether my values are null

            See Also
            --------
            isna
            isnull
            Categorical.notna

            """
        
        
        ```
        ''',
        "good_example": '''```python
        
        def isna(self):
            """
            Detect missing values

            Missing values (-1 in .codes) are detected.

            Returns
            -------
            a boolean array of whether my values are null

            See Also
            --------
            isna : Top-level isna.
            isnull : Alias of isna.
            Categorical.notna : Boolean inverse of Categorical.isna.

            """
        
        ```
        '''
    },
    "EX01": {
        "description": "No examples section found",
        "references": "",
        "bad_example": '''```python
        
        @staticmethod
        def _run_os(*args):
            """
            Execute a command as a OS terminal.

            Parameters
            ----------
            *args : list of str
                Command and parameters to be executed
            """
        
        
        ```
        ''',
        "good_example": '''```python
        
        @staticmethod
        def _run_os(*args):
            """
            Execute a command as a OS terminal.

            Parameters
            ----------
            *args : list of str
                Command and parameters to be executed

            Examples
            --------
            >>> DocBuilder()._run_os('python', '--version')
            """
        
        ```
        '''
    },
    
    
    "GL0Z": {
        "description": "",
        "references": "- [Flake8 rules](https://www.flake8rules.com/)",
        "bad_example": '''```python
        
        
        ```
        ''',
        "good_example": '''```python
        
        
        ```
        '''
    }
}