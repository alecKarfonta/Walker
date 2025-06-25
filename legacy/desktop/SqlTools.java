package com.alec.walker.desktop;

import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Type;
import java.sql.ResultSet;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

public class SqlTools {

	static boolean					isDebug			= true;
	static Gson						gson			= new Gson();
	private static SimpleDateFormat	dateFormat		= new SimpleDateFormat("yyyy-MM-dd");
	private static Type				stringListType	= new TypeToken<ArrayList<String>>() {
													}.getType();

	public static DatabaseObject sqlParse(DatabaseObject resultObject,
			ResultSet resultSet) {
		if (isDebug) {
			System.out.println("sqlParse()");
		}
		try {
			//

			// for each property of the actual object type
			for (PropertyDescriptor pd : Introspector.getBeanInfo(
					resultObject.getClass()).getPropertyDescriptors()) {

				String propertyName = pd.getDisplayName();

				// if the property has a write method
				if (pd.getWriteMethod() != null) { // && !"class".equals(pd.getName())) {

					String propertyType = pd.getPropertyType().toString();
					propertyType = propertyType.substring(propertyType.lastIndexOf(".") + 1,
							propertyType.length());

					// if it is the id property
					if (propertyName.equalsIgnoreCase("id")) {
						// set it
						pd.getWriteMethod().invoke(resultObject, resultSet.getInt("id"));

						// else try to match the property to a field
					} else {
						try {

							// then use the setter to fill the object's value
							switch (propertyType) {
								case "String":
									pd.getWriteMethod().invoke(resultObject,
											resultSet.getString(propertyName));
									break;
								case "int":
									pd.getWriteMethod().invoke(resultObject,
											resultSet.getInt(propertyName));
									break;
								case "float":
									propertyName = propertyName.substring(0, 1).toUpperCase()
											+ propertyName.substring(1);
									pd.getWriteMethod().invoke(resultObject,
											resultSet.getFloat(propertyName));
									break;
								case "boolean":
									propertyName = propertyName.substring(0, 1).toUpperCase()
											+ propertyName.substring(1);
									pd.getWriteMethod().invoke(resultObject,
											resultSet.getBoolean(propertyName));
									break;
								// if object is array list
								case "ArrayList":
									// then parse the object as a string array
									pd.getWriteMethod()
											.invoke(
													resultObject,
													gson.fromJson(
															resultSet.getString(propertyName),
															stringListType
															)
											);
									break;
							}
						} catch (Exception e) {
							e.printStackTrace();
						}
					}
				}
			}

			return resultObject;

		} catch (Exception e2) {
			e2.printStackTrace();
		}

		return null;
	}

	public static String where(DatabaseObject searchObject) {
		String sql = "";
		// extract the class name out of the full package path
		String className = searchObject.getClass().getName();
		className = className.substring(className.lastIndexOf(".") + 1, className.length());
		boolean isFirstParam = true;
		Field[] fields = searchObject.getClass().getDeclaredFields();

		for (Field field : fields) {
			try {
				String fieldName = field.getName();
				// get the value of the property
				Object fieldValue = PropertyUtils.getSimpleProperty(searchObject, fieldName);

				if (fieldValue != null) {
					// extract the actual type from the object
					String actualType = fieldValue.getClass().getName();
					// the class name starts after the last '.'
					actualType = actualType.substring(actualType.lastIndexOf(".") + 1,
							actualType.length());

					// skip on the following values
					// if the field type is arraylist
					if (actualType.matches("ArrayList")
							// or if its an empty string
							|| (actualType.matches("String") && ((String) fieldValue).matches(""))
							// or if its an empty integer
							|| (actualType.matches("Integer") && ((Integer) fieldValue).intValue() == 0)
							// or if its an empty float
							|| (actualType.matches("Float") && ((Float) fieldValue).floatValue() == 0.0)
							// or it its a false boolean
							|| (actualType.matches("Boolean") && ((Boolean) fieldValue) == false)) {
						continue;
					}
					// if this is the first search parameter
					if (isFirstParam) {
						// next time it's not
						isFirstParam = false;
					} else {
						// add the AND between WHERE clauses
						sql += " AND ";
					}

					// add column name to sql
					sql += " `" + fieldName + "`";

					// add the value of the search parameter based on the type;
					// strings are wrapped in '
					// int and bool have no quotes
					if (actualType.matches("String")) {
						sql += " LIKE '" + fieldValue + "' ";
					} else if (actualType.matches("Timestamp")) {
						Timestamp timestamp = (Timestamp) fieldValue;
						sql += " = '" + dateFormat.format(timestamp) + "'";
					} else {
						sql += " = " + fieldValue;
					}
				}

			} catch (IllegalAccessException e) {
				e.printStackTrace();
			} catch (InvocationTargetException e) {
				e.printStackTrace();
			} catch (NoSuchMethodException e) {
				e.printStackTrace();
			}

		}

		return sql;
	}

	public static String sqlSearch(DatabaseObject searchObject) {
		if (isDebug)
		{
			System.out.println("MySqlAccess.sqlSearch(" + gson.toJson(searchObject) + ")");
		}
		// extact the class name out of the full package path
		String className = searchObject.getClass().getName();
		// classname starts one char after the last '.' in the package path
		className = className.substring(className.lastIndexOf(".") + 1, className.length());

		// start the sql
		String sql = "SELECT * FROM `" + className + "` WHERE ";

		Field[] fields = searchObject.getClass().getDeclaredFields();

		boolean isFirstParam = true;

		// if the id is defined
		try {
			int id = (Integer) PropertyUtils.getSimpleProperty(searchObject, "id");
			if (id != 0) {
				isFirstParam = false;
				sql += "`id` = " + id;
			}
		} catch (IllegalAccessException e1) {
			e1.printStackTrace();
		} catch (InvocationTargetException e1) {
			e1.printStackTrace();
		} catch (NoSuchMethodException e1) {
			e1.printStackTrace();
		}

		// for each propery of the databaseObject's actual type
		for (Field field : fields) {

			// add a key-value pair for the field
			try {	// try to get the property value from the databaseObject using the field name

				String fieldName = field.getName();
				// get the value of the property
				Object fieldValue = PropertyUtils.getSimpleProperty(searchObject, fieldName);

				if (fieldValue != null) {

					// extract the actual type from the object
					String actualType = fieldValue.getClass().getName();
					// the class name starts after the last '.'
					actualType = actualType.substring(actualType.lastIndexOf(".") + 1,
							actualType.length());

					// // extract the actual type from the object
					// String actualType = fieldValue.getClass().getName();
					// // the class name starts after the last '.'
					// actualType = actualType.substring(actualType.lastIndexOf(".") + 1, actualType.length());

					// if the field type is arraylist
					if (actualType.matches("ArrayList")) {
						// TODO: handle arraylists stored as json in the db
						if (isDebug) {
							System.out.println("Skip Arraylist");
						}
						continue;
						// // assume it's stored in the db as a json string
						// String gString = gson.toJson(fieldValue);
						// // if it's empty
						// if (((ArrayList)fieldValue).size() == 0 && gString.matches("")) {
						// // skip
						// continue;
						// }

					}

					// ignore null values for each data type
					if (actualType.matches("String")) {
						if (((String) fieldValue).matches("")) {
							if (isDebug) {
								System.out.println("Skip Empty String");
							}
							continue;
						}
					}
					if (actualType.matches("Integer")) {
						if (((Integer) fieldValue).intValue() == 0) {
							if (isDebug) {
								System.out.println("Skip 0 int");
							}
							continue;
						}
					}
					if (actualType.matches("Float")) {
						if (((Float) fieldValue).floatValue() == 0.0) {
							if (isDebug) {
								System.out.println("Skip 0.0 float");
							}
							continue;
						}
					}
					if (actualType.matches("Boolean")) {
						if (!((Boolean) fieldValue)) {
							if (isDebug) {
								System.out.println("Skip false bool");
							}
							continue;
						}
					}

					// if this is the first search parameter
					if (isFirstParam) {
						// next time it's not
						isFirstParam = false;
					} else {
						// add the AND between WHERE clauses
						sql += " AND ";
					}

					// add column name to sql
					sql += " `" + fieldName + "`";

					// add the value of the search parameter based on the type;
					// strings are wrapped in '
					// int and bool have no quotes
					if (actualType.matches("String")) {
						sql += " LIKE '" + fieldValue + "' ";
					} else if (actualType.matches("Timestamp")) {
						Timestamp timestamp = (Timestamp) fieldValue;
						sql += " = '" + dateFormat.format(timestamp) + "'";
					} else {
						sql += " = " + fieldValue;
					}

					if (isDebug) {
						System.out.println();
						System.out.println("################### Field ######################");
						System.out.println("");

						System.out.println("Name: " + fieldName);
						System.out.println("Value: " + fieldValue);
						System.out.println("Type: " + actualType);

						System.out.println("################################################");
						System.out.println();
					}

				}

			} catch (IllegalAccessException e) {
				e.printStackTrace();
			} catch (InvocationTargetException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (NoSuchMethodException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}

		// clear the WHERE if no params were added
		if (isFirstParam) {
			sql = sql.replace("WHERE", "");
		}
		if (isDebug) {
			System.out.println("MySqlAccess.sqlSearch() : Sql generated = " + sql);
		}
		return sql;

	}

	public static String update(DatabaseObject databaseObject) {
		System.out.println("SqlTools.update( " + gson.toJson(databaseObject) + ")");
		// extact the class name out of the full package path
		String className = databaseObject.getClass().getName();
		// classname starts one char after the last '.' in the package path
		className = className.substring(className.lastIndexOf(".") + 1, className.length());

		// start the sql
		String sql = "UPDATE `" + className + "` SET ";

		Field[] fields = databaseObject.getClass().getDeclaredFields();

		// for each propery of the databaseObject's actual type
		for (Field field : fields) {

			// add a key-value pair for the field
			try {	// try to get the property value from the databaseObject using the field name

				String fieldName = field.getName();
				// get the value of the property
				Object fieldValue = PropertyUtils.getSimpleProperty(databaseObject, fieldName);

				if (fieldValue == null) {
					continue;
				}

				// extract the actual type from the object
				String actualType = fieldValue.getClass().getName();

				actualType = actualType.substring(actualType.lastIndexOf(".") + 1,
						actualType.length());

				// if in debug mode show the output
				if (isDebug) {
					System.out.println();
					System.out.println("################### Field ######################");
					System.out.println("");

					System.out.println("Name: " + fieldName);
					System.out.println("Value: " + fieldValue);
					System.out.println("Type: " + actualType);

					System.out.println("################################################");
					System.out.println();
				}

				// the class name starts after the last '.'
				if (actualType == null || actualType.equalsIgnoreCase("ArrayList")) {
					continue;
				}

				switch (actualType) {
					case "Integer":
						// then if values is 0
						if ((Integer) fieldValue == 0) {
							// skip
							continue;
						}
						break;
					case "String":
						// if empty
						if (((String) fieldValue).isEmpty()) {
							continue;
						}
				}

				// don't update id's
				if (fieldName.matches("id")) {
					continue;
				}

				sql += "`" + fieldName + "`";
				sql += " = ";

				// if value is a string add quotes
				if (actualType.matches("String")) {
					sql += "'" + PropertyUtils.getSimpleProperty(databaseObject, fieldName) + "' ";
				} else if (actualType.matches("Timestamp")) {
					Timestamp timestamp = (Timestamp) fieldValue;
					Calendar cal = Calendar.getInstance();
					cal.setTime(timestamp);
					// cal.add(Calendar.DAY_OF_WEEK, -1);
					timestamp.setTime(cal.getTime().getTime()); // or
					// timestamp = new Timestamp(cal.getTime().getTime());
					sql += "'" + dateFormat.format(timestamp) + "'";
				} else {
					sql += PropertyUtils.getSimpleProperty(databaseObject, fieldName);
				}

				sql += ", ";

			} catch (IllegalAccessException | InvocationTargetException
					| NoSuchMethodException e) {
				e.printStackTrace();
			}

		}

		if (sql.endsWith(", ")) {
			sql = sql.substring(0, sql.length() - 2);
		}
		// add the where clause to match id's
		try {
			sql += " WHERE `id` = " + PropertyUtils.getSimpleProperty(databaseObject, "id");
		} catch (IllegalAccessException | InvocationTargetException
				| NoSuchMethodException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return sql;

	}

	public static String sqlInsert(String db, DatabaseObject databaseObject) {

		// extact the class name out of the full package path
		String className = databaseObject.getClass().getName();
		// classname starts one char after the last '.' in the package path
		className = className.substring(className.lastIndexOf(".") + 1, className.length());

		// start the sql
		String sql = "INSERT INTO `" + db + "`.`" + className + "` (";

		Field[] fields = databaseObject.getClass().getDeclaredFields();
		int fieldCount = fields.length;
		int currentField = 0;

		// build the list of column names
		// for each propery of the databaseObject's actual type
		for (Field field : fields) {
			// add a key-value pair for the field
			String fieldName = field.getName();

			sql += "`" + fieldName + "`";

			// if it's not the last field
			if (currentField < fields.length - 1) {
				// add a comma
				sql += ", ";
			}

			// next field
			currentField++;
		}

		// more SQL
		sql += ") VALUES ( ";

		currentField = 0;
		for (Field field : fields) {
			// get the value of the property
			try {
				String fieldName = field.getName();
				Object fieldValue = PropertyUtils.getSimpleProperty(databaseObject, fieldName);

				// if the field is a string
				if (field.getType().toString().contains("String")) {
					// if the value is not null
					if (fieldValue != null) {
						sql += "'" + PropertyUtils.getSimpleProperty(databaseObject, fieldName)
								+ "' ";
					} else {
						sql += "'' ";
					}
				} else {
					sql += PropertyUtils.getSimpleProperty(databaseObject, fieldName);
				}

				// if not the last element, append a ','
				if (currentField < fieldCount - 1) {
					sql += ", ";
				}
				// next field
				currentField++;
			} catch (IllegalAccessException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InvocationTargetException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (NoSuchMethodException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
		sql += ");";

		return sql;

	}

}
